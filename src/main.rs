use std::fs;
use std::env;
use petgraph::dot::{Dot, Config};
use petgraph::graph::{NodeIndex};
use petgraph::Graph;
use petgraph::Directed;
use std::process::Command;
use std::collections::HashSet;

// Fabian Prado - Sofia Lopez
// Laboratorio 4 - Teoría de la Computación

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Literal(char),
    CharClass(Vec<char>),
    Alternation,
    Concatenation,
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
    LeftParen,
    RightParen,
}

#[derive(Debug)]
pub enum Error {
    Parse,
    File,
}

pub struct Tokenizer {
    chars: Vec<char>,
    pos: usize,
}

impl Tokenizer {
    pub fn new(input: &str) -> Self {
        Self { chars: input.chars().collect(), pos: 0 }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, Error> {
        let mut tokens = Vec::new();
        
        while self.pos < self.chars.len() {
            match self.chars[self.pos] {
                '\\' => {
                    self.pos += 1;
                    if self.pos < self.chars.len() {
                        tokens.push(Token::Literal(self.chars[self.pos]));
                    }
                }
                '[' => {
                    self.pos += 1;
                    let mut chars = Vec::new();
                    while self.pos < self.chars.len() && self.chars[self.pos] != ']' {
                        chars.push(self.chars[self.pos]);
                        self.pos += 1;
                    }
                    tokens.push(Token::CharClass(chars));
                }
                '|' => tokens.push(Token::Alternation),
                '*' => tokens.push(Token::ZeroOrMore),
                '+' => tokens.push(Token::OneOrMore),
                '?' => tokens.push(Token::ZeroOrOne),
                '(' => tokens.push(Token::LeftParen),
                ')' => tokens.push(Token::RightParen),
                'E' => tokens.push(Token::Literal('ε')),
                c => tokens.push(Token::Literal(c)),
            }
            self.pos += 1;
        }

        // Insert concatenation
        let mut result = Vec::new();
        for i in 0..tokens.len() {
            if i > 0 && self.needs_concat(&tokens[i-1], &tokens[i]) {
                result.push(Token::Concatenation);
            }
            result.push(tokens[i].clone());
        }
        
        Ok(result)
    }

    fn needs_concat(&self, prev: &Token, curr: &Token) -> bool {
        matches!(prev, Token::Literal(_) | Token::CharClass(_) | Token::ZeroOrMore | Token::OneOrMore | Token::ZeroOrOne | Token::RightParen) &&
        matches!(curr, Token::Literal(_) | Token::CharClass(_) | Token::LeftParen)
    }
}

pub struct Parser {
    tokens: Vec<Token>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens}
    }

    pub fn parse(&mut self) -> Result<Vec<Token>, Error> {
        let mut output = Vec::new();
        let mut stack = Vec::new();

        for token in &self.tokens {
            match token {
                Token::Literal(_) | Token::CharClass(_) => output.push(token.clone()),
                Token::ZeroOrMore | Token::OneOrMore | Token::ZeroOrOne => output.push(token.clone()),
                Token::LeftParen => stack.push(token.clone()),
                Token::RightParen => {
                    while let Some(op) = stack.pop() {
                        if matches!(op, Token::LeftParen) { break; }
                        output.push(op);
                    }
                }
                Token::Concatenation => {
                    while let Some(top) = stack.last() {
                        if matches!(top, Token::LeftParen) { break; }
                        output.push(stack.pop().unwrap());
                    }
                    stack.push(token.clone());
                }
                Token::Alternation => {
                    while let Some(top) = stack.last() {
                        if matches!(top, Token::LeftParen) { break; }
                        output.push(stack.pop().unwrap());
                    }
                    stack.push(token.clone());
                }
            }
        }

        while let Some(op) = stack.pop() {
            output.push(op);
        }

        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct TreeNode {
    pub token: Token,
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    pub fn new(token: Token) -> Self {
        Self {
            token,
            children: Vec::new(),
        }
    }
    
    pub fn new_with_children(token: Token, children: Vec<TreeNode>) -> Self {
        Self {
            token,
            children,
        }
    }
}

pub struct SyntaxTreeBuilder {
    postfix: Vec<Token>,
}

impl SyntaxTreeBuilder {
    pub fn new(postfix: Vec<Token>) -> Self {
        Self { postfix }
    }
    
    pub fn build_tree(&self) -> Result<TreeNode, Error> {
        let mut stack: Vec<TreeNode> = Vec::new();
        
        for token in &self.postfix {
            match token {
                Token::Literal(_) | Token::CharClass(_) => {
                    stack.push(TreeNode::new(token.clone()));
                }
                Token::ZeroOrMore | Token::OneOrMore | Token::ZeroOrOne => {
                    if let Some(operand) = stack.pop() {
                        let node = TreeNode::new_with_children(token.clone(), vec![operand]);
                        stack.push(node);
                    }
                }
                Token::Concatenation | Token::Alternation => {
                    if stack.len() >= 2 {
                        let right = stack.pop().unwrap();
                        let left = stack.pop().unwrap();
                        let node = TreeNode::new_with_children(token.clone(), vec![left, right]);
                        stack.push(node);
                    }
                }
                _ => {} 
            }
        }
        
        stack.pop().ok_or(Error::Parse)
    }
}

pub struct TreeVisualizer {
    graph: Graph<String, (), Directed>,
}

impl TreeVisualizer {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }
    
    fn token_to_label(&self, token: &Token) -> String {
        match token {
            Token::Literal(c) => c.to_string(),
            Token::CharClass(chars) => format!("[{}]", chars.iter().collect::<String>()),
            Token::Alternation => "|".to_string(),
            Token::Concatenation => "·".to_string(),
            Token::ZeroOrMore => "*".to_string(),
            Token::OneOrMore => "+".to_string(),
            Token::ZeroOrOne => "?".to_string(),
            _ => "?".to_string(),
        }
    }
    
    fn add_tree_nodes(&mut self, tree: &TreeNode) -> NodeIndex {
        let label = self.token_to_label(&tree.token);
        let node_index = self.graph.add_node(label);
        
        for child in &tree.children {
            let child_index = self.add_tree_nodes(child);
            self.graph.add_edge(node_index, child_index, ());
        }
        
        node_index
    }
    
    pub fn visualize_tree(&mut self, tree: &TreeNode, filename: &str) -> Result<(), Error> {
        self.add_tree_nodes(tree);
        
        let dot_output = format!("{:?}", Dot::with_config(&self.graph, &[Config::EdgeNoLabel]));
        
        fs::write(filename, dot_output).map_err(|_| Error::File)?;
        
        println!("Árbol guardado en: {}", filename);
        println!("Para visualizarlo, usa: dot -Tpng {} -o tree.png", filename);
        
        Ok(())
    }
    
    pub fn print_tree(&self, tree: &TreeNode) {
        println!("Árbol sintáctico:");
        self.print_tree_recursive(tree, "", true);
        println!();
    }
    
    fn print_tree_recursive(&self, node: &TreeNode, prefix: &str, is_last: bool) {
        let connector = if is_last { "└── " } else { "├── " };
        let label = self.token_to_label(&node.token);
        
        println!("{}{}{}", prefix, connector, label);
        
        let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
        
        for (i, child) in node.children.iter().enumerate() {
            let is_last_child = i == node.children.len() - 1;
            self.print_tree_recursive(child, &new_prefix, is_last_child);
        }
    }
}

#[derive(Debug, Clone)]
pub struct NFATransition {
    pub symbol: Option<char>,  // None represents epsilon 
    pub to_state: usize,       // ID of destination 
}

#[derive(Debug, Clone)]
pub struct NFAState {
    pub id: usize,
    pub is_final: bool,
    pub transitions: Vec<NFATransition>,
}

#[derive(Debug, Clone)]
pub struct NFA {
    pub states: Vec<NFAState>,
    pub start_state: usize,
    pub final_state: usize,
}

impl NFAState {
    pub fn new(id: usize, is_final: bool) -> Self {
        Self {
            id,
            is_final,
            transitions: Vec::new(),
        }
    }
    
    pub fn add_transition(&mut self, symbol: Option<char>, to_state: usize) {
        self.transitions.push(NFATransition { symbol, to_state });
    }
}

impl NFA {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            start_state: 0,
            final_state: 0,
        }
    }
    
    // Helper to create a 2-state NFA for a character
    pub fn from_char(c: char, state_counter: &mut usize) -> Self {
        let start_id = *state_counter;
        let final_id = *state_counter + 1;
        *state_counter += 2;
        
        let mut start_state = NFAState::new(start_id, false);
        start_state.add_transition(Some(c), final_id);
        
        let final_state = NFAState::new(final_id, true);
        
        Self {
            states: vec![start_state, final_state],
            start_state: start_id,
            final_state: final_id,
        }
    }
    
    // Helper to create NFA for epsilon
    pub fn from_epsilon(state_counter: &mut usize) -> Self {
        let start_id = *state_counter;
        let final_id = *state_counter + 1;
        *state_counter += 2;
        
        let mut start_state = NFAState::new(start_id, false);
        start_state.add_transition(None, final_id);  // None = epsilon
        
        let final_state = NFAState::new(final_id, true);
        
        Self {
            states: vec![start_state, final_state],
            start_state: start_id,
            final_state: final_id,
        }
    }
}

pub struct ThompsonBuilder {
    state_counter: usize,
}

impl ThompsonBuilder {
    pub fn new() -> Self {
        Self { state_counter: 0 }
    }
    
    pub fn build_from_tree(&mut self, tree: &TreeNode) -> NFA {
        match &tree.token {
            Token::Literal(c) => {
                if *c == 'ε' {
                    self.build_epsilon()
                } else {
                    self.build_literal(*c)
                }
            }
            Token::Concatenation => {
                let left = self.build_from_tree(&tree.children[0]);
                let right = self.build_from_tree(&tree.children[1]);
                self.concatenate(left, right)
            }
            Token::Alternation => {
                let left = self.build_from_tree(&tree.children[0]);
                let right = self.build_from_tree(&tree.children[1]);
                self.alternate(left, right)
            }
            Token::ZeroOrMore => {
                let child = self.build_from_tree(&tree.children[0]);
                self.kleene_star(child)
            }
            Token::OneOrMore => {
                let child = self.build_from_tree(&tree.children[0]);
                self.one_or_more(child)
            }
            Token::ZeroOrOne => {
                let child = self.build_from_tree(&tree.children[0]);
                self.zero_or_one(child)
            }
            _ => panic!("Invalid token: {:?}", tree.token),
        }
    }
    
    // Build NFA for single character
    fn build_literal(&mut self, c: char) -> NFA {
        let start = self.state_counter;
        let end = self.state_counter + 1;
        self.state_counter += 2;
        
        let mut states = vec![
            NFAState::new(start, false),
            NFAState::new(end, true),
        ];
        
        states[0].add_transition(Some(c), end);
        
        NFA {
            states,
            start_state: start,
            final_state: end,
        }
    }
    
    // Build NFA for epsilon
    fn build_epsilon(&mut self) -> NFA {
        let start = self.state_counter;
        let end = self.state_counter + 1;
        self.state_counter += 2;
        
        let mut states = vec![
            NFAState::new(start, false),
            NFAState::new(end, true),
        ];
        
        states[0].add_transition(None, end);  // None = epsilon
        
        NFA {
            states,
            start_state: start,
            final_state: end,
        }
    }
    
    // Concatenation
    fn concatenate(&mut self, mut left: NFA, mut right: NFA) -> NFA {
        // Connect left to right with epsilon
        for state in &mut left.states {
            if state.id == left.final_state {
                state.add_transition(None, right.start_state);
                state.is_final = false;  
            }
        }

        // Update final states to connect to new end
        let mut all_states = left.states;
        all_states.extend(right.states);

        NFA {
            states: all_states,
            start_state: left.start_state,
            final_state: right.final_state,
        }
    }
    
    // Alternation: One or the other
    fn alternate(&mut self, left: NFA, right: NFA) -> NFA {
        // Create new start and end states
        let new_start = self.state_counter;
        let new_end = self.state_counter + 1;
        self.state_counter += 2;
        
        let mut states = vec![
            NFAState::new(new_start, false),
            NFAState::new(new_end, true),
        ];
        
        // New start connects to both paths
        states[0].add_transition(None, left.start_state);
        states[0].add_transition(None, right.start_state);
        
        // Add all states from left and right
        states.extend(left.states);
        states.extend(right.states);
        
        // Update final states to connect to new end
        for state in &mut states {
            if state.id == left.final_state || state.id == right.final_state {
                state.add_transition(None, new_end);
                state.is_final = false;
            }
        }
        
        NFA {
            states,
            start_state: new_start,
            final_state: new_end,
        }
    }
    
    // Kleene star: zero or more
    fn kleene_star(&mut self, inner: NFA) -> NFA {
        // Create new start and end states
        let new_start = self.state_counter;
        let new_end = self.state_counter + 1;
        self.state_counter += 2;
        
        let mut states = vec![
            NFAState::new(new_start, false),
            NFAState::new(new_end, true),
        ];
        
        // Epsilon from new start to original start
        states[0].add_transition(None, inner.start_state);
        // Epsilon from new start to new end 
        states[0].add_transition(None, new_end);
        
        states.extend(inner.states);
        
        // Update final states to connect to new end and loop back
        for state in &mut states {
            if state.id == inner.final_state {
                state.add_transition(None, inner.start_state);  // loop
                state.add_transition(None, new_end);            
                state.is_final = false;
            }
        }
        
        NFA {
            states,
            start_state: new_start,
            final_state: new_end,
        }
    }
    
    // Plus: one or more
    fn one_or_more(&mut self, inner: NFA) -> NFA {
        // Create new start and end states
        let new_start = self.state_counter;
        let new_end = self.state_counter + 1;
        self.state_counter += 2;
        
        let mut states = vec![
            NFAState::new(new_start, false),
            NFAState::new(new_end, true),
        ];
        
        // Epsilon from new start to original start
        states[0].add_transition(None, inner.start_state);
        
        states.extend(inner.states);
        
        // Update final states to connect to new end and loop back
        for state in &mut states {
            if state.id == inner.final_state {
                state.add_transition(None, inner.start_state);  // loop
                state.add_transition(None, new_end);           
                state.is_final = false;
            }
        }
        
        NFA {
            states,
            start_state: new_start,
            final_state: new_end,
        }
    }
    
    // Question: zero or one
    fn zero_or_one(&mut self, inner: NFA) -> NFA {
        // Create new start and end states
        let new_start = self.state_counter;
        let new_end = self.state_counter + 1;
        self.state_counter += 2;
        
        let mut states = vec![
            NFAState::new(new_start, false),
            NFAState::new(new_end, true),
        ];
        
        // Epsilon from new start to original start
        states[0].add_transition(None, inner.start_state);

        // Epsilon from new start to new end
        states[0].add_transition(None, new_end);
        
        states.extend(inner.states);
        
        // Update final states to connect to new end
        for state in &mut states {
            if state.id == inner.final_state {
                state.add_transition(None, new_end);
                state.is_final = false;
            }
        }
        
        NFA {
            states,
            start_state: new_start,
            final_state: new_end,
        }
    }
}

// For its simplistic approach, we will use graphiz to visualize the NFA
// brew instal graphviz (run this if u dont have it installed)
pub struct NFAVisualizer; 

impl NFAVisualizer {
    pub fn to_dot(nfa: &NFA) -> String {
        let mut dot = String::new();
        dot.push_str("digraph NFA {\n");
        dot.push_str("    rankdir=LR;\n");
        dot.push_str("    node [shape=circle];\n");
        
        // Mark initial state with incoming arrow
        dot.push_str(&format!("    start [shape=none, label=\"\"];\n"));
        dot.push_str(&format!("    start -> {};\n", nfa.start_state));
        
        // Mark final state with double circle
        dot.push_str(&format!("    {} [shape=doublecircle];\n", nfa.final_state));
        
        // Add all transitions
        for state in &nfa.states {
            for transition in &state.transitions {
                let label = match transition.symbol {
                    Some(c) => c.to_string(),
                    None => "ε".to_string(),
                };
                dot.push_str(&format!("    {} -> {} [label=\"{}\"];\n", 
                    state.id, transition.to_state, label));
            }
        }
        
        dot.push_str("}\n");
        dot
    }
    
    pub fn save_to_file(nfa: &NFA, filename: &str) -> Result<(), Error> {
        let dot_content = Self::to_dot(nfa);
        fs::write(filename, &dot_content).map_err(|_| Error::File)?;
        
        // generate png
        let png_filename = filename.replace(".dot", ".png");
        let output = Command::new("dot")
            .args(&["-Tpng", filename, "-o", &png_filename])
            .output();
        
        match output {
            Ok(_) => {
                println!("NFA guardado en: {}", filename);
                println!("PNG generado en: {}", png_filename);
            }
            Err(_) => {
                println!("NFA guardado en: {}", filename);
                println!("Error generando PNG (brew install graphviz)");
            }
        }
        
        Ok(())
    }

    pub fn print_nfa_details(nfa: &NFA) {
        println!("\n===================");
        println!("NFA");
        println!("Start: {}", nfa.start_state);
        println!("Final: {}", nfa.final_state);
        println!("\nTransitions:");
        
        for state in &nfa.states {
            for transition in &state.transitions {
                let symbol = match transition.symbol {
                    Some(c) => c.to_string(),
                    None => "ε".to_string(),
                };
                println!("  State {} --[{}]--> State {}", 
                    state.id, symbol, transition.to_state);
            }
        }
        println!("===================\n");
    }
}

pub struct NFASimulator;

impl NFASimulator {
    fn epsilon_closure(nfa: &NFA, states: HashSet<usize>) -> HashSet<usize> {
        let mut closure = states.clone();
        let mut stack: Vec<usize> = states.into_iter().collect();
        
        while let Some(state_id) = stack.pop() {
            if let Some(state) = nfa.states.iter().find(|s| s.id == state_id) {
                for transition in &state.transitions {
                    if transition.symbol.is_none() {
                        if closure.insert(transition.to_state) {
                            stack.push(transition.to_state);
                        }
                    }
                }
            }
        }
        closure
    }
    
    fn move_states(nfa: &NFA, states: &HashSet<usize>, symbol: char) -> HashSet<usize> {
        let mut next_states = HashSet::new();
        
        for &state_id in states {
            if let Some(state) = nfa.states.iter().find(|s| s.id == state_id) {
                for transition in &state.transitions {
                    if let Some(trans_symbol) = transition.symbol {
                        if trans_symbol == symbol {
                            next_states.insert(transition.to_state);
                        }
                    }
                }
            }
        }
        next_states
    }
    
    pub fn simulate(nfa: &NFA, input: &str) -> bool {
        let mut current_states = HashSet::new();
        current_states.insert(nfa.start_state);
        current_states = Self::epsilon_closure(nfa, current_states);
        
        for ch in input.chars() {
            let next_states = Self::move_states(nfa, &current_states, ch);
            current_states = Self::epsilon_closure(nfa, next_states);
            
            if current_states.is_empty() {
                return false;
            }
        }
        
        current_states.contains(&nfa.final_state)
    }
}

fn process_regex(input: &str, index: usize) -> Result<NFA, Error> {
    println!("Input: {}", input);
    
    let mut tokenizer = Tokenizer::new(input);
    let tokens = tokenizer.tokenize()?;
    
    let mut parser = Parser::new(tokens);
    let postfix = parser.parse()?;
    
    println!("Postfix: {:?}", postfix);
    
    let tree_builder = SyntaxTreeBuilder::new(postfix);
    let syntax_tree = tree_builder.build_tree()?;
    
    let visualizer = TreeVisualizer::new();
    visualizer.print_tree(&syntax_tree);

    let mut thompson = ThompsonBuilder::new();
    let nfa = thompson.build_from_tree(&syntax_tree);
    
    NFAVisualizer::print_nfa_details(&nfa);
    
    let filename = format!("nfa_{}.dot", index);
    NFAVisualizer::save_to_file(&nfa, &filename)?;
    
    Ok((nfa))
}

// Precheck if a string contains regex operators
fn is_regex(s: &str) -> bool {
    s.contains('*') || s.contains('+') || s.contains('?') || 
    s.contains('|') || s.contains('(') || s.contains(')')
}
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage: {} <input.txt>", args[0]);
        return;
    }

    let file_path = &args[1];
    
    let current_dir = env::current_dir().unwrap();
    println!("Directorio actual: {:?}", current_dir);
    println!("Buscando archivo: {:?}", file_path);
    
    let content = match fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(e) => {
            println!("Error leyendo archivo '{}': {}", file_path, e);
            println!("Asegúrate de que el archivo existe en: {:?}", current_dir.join(file_path));
            return;
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    let mut regex_count = 0;
    
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() {
            i += 1;
            continue;
        }
        
        if is_regex(line) {
            regex_count += 1;
            println!("\nProcesando expresión regular {}", regex_count);
            
            match process_regex(line, regex_count) {
                Ok(nfa) => {
                    i += 1;
                    while i < lines.len() && !is_regex(lines[i]) && !lines[i].trim().is_empty() {
                        let test_string = lines[i].trim();
                        
                        let result = NFASimulator::simulate(&nfa, test_string);
                        println!("w = '{}': {}", test_string, if result { "sí" } else { "no" });
                        
                        i += 1;
                    }
                }
                Err(e) => {
                    println!("Error procesando '{}': {:?}", line, e);
                    i += 1;
                }
            }
        } else {
            println!("Línea inválida ignorada: '{}'", line);
            i += 1;
        }
    }
}