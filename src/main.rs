use std::fs;
use std::env;
use petgraph::dot::{Dot, Config};
use petgraph::graph::{NodeIndex};
use petgraph::Graph;
use petgraph::Directed;

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

fn process_regex(input: &str) -> Result<(), Error> {
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
    
    Ok(())
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
    
    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        
        println!("--- Procesando línea {} ---", index + 1);
        if let Err(e) = process_regex(line) {
            println!("Error procesando '{}': {:?}", line, e);
        }
        println!(); 
    }
}