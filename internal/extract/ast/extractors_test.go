package ast

import (
	"context"
	"os"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func TestPythonExtractor(t *testing.T) {
	src := `import os
from pathlib import Path

def greet(name):
    return format_name(name)

def format_name(name):
    return name.upper()

class Animal:
    def speak(self):
        pass
`
	f := writeTempFile(t, "test.py", src)
	nodes, edges, err := (&PythonExtractor{}).ExtractFile(context.Background(), f)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	logNodesEdges(t, nodes, edges)

	assertLabel(t, nodes, "greet")
	assertLabel(t, nodes, "format_name")
	assertLabel(t, nodes, "Animal")
	assertLabel(t, nodes, "Animal.speak")
	assertEdgeKind(t, edges, "IMPORTS")
	assertEdgeKind(t, edges, "CALLS")
}

func TestJavaScriptExtractor(t *testing.T) {
	src := `import { readFile } from 'fs';
import path from 'path';

function process(data) {
    return transform(data);
}

function transform(data) {
    return data.trim();
}

class Parser {
    parse(input) {
        return process(input);
    }
}
`
	f := writeTempFile(t, "test.js", src)
	nodes, edges, err := (&JavaScriptExtractor{}).ExtractFile(context.Background(), f)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	logNodesEdges(t, nodes, edges)

	assertLabel(t, nodes, "process")
	assertLabel(t, nodes, "transform")
	assertLabel(t, nodes, "Parser")
	assertEdgeKind(t, edges, "IMPORTS")
}

func TestRustExtractor(t *testing.T) {
	src := `use std::fs;
use std::io::Result;

fn read_file(path: &str) -> Result<String> {
    let content = parse_content(path);
    Ok(content)
}

fn parse_content(path: &str) -> String {
    path.to_string()
}

struct Config {
    path: String,
}

impl Config {
    fn new(path: &str) -> Self {
        Config { path: path.to_string() }
    }
}
`
	f := writeTempFile(t, "test.rs", src)
	nodes, edges, err := (&RustExtractor{}).ExtractFile(context.Background(), f)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	logNodesEdges(t, nodes, edges)

	assertLabel(t, nodes, "read_file")
	assertLabel(t, nodes, "parse_content")
	assertLabel(t, nodes, "Config")
	assertLabel(t, nodes, "Config.new")
	assertEdgeKind(t, edges, "IMPORTS")
}

func TestJavaExtractor(t *testing.T) {
	src := `package com.example;

import java.util.List;
import java.util.ArrayList;

public class Processor {
    public String process(String input) {
        return format(input);
    }

    private String format(String s) {
        return s.trim();
    }
}
`
	f := writeTempFile(t, "Processor.java", src)
	nodes, edges, err := (&JavaExtractor{}).ExtractFile(context.Background(), f)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	logNodesEdges(t, nodes, edges)

	assertLabel(t, nodes, "Processor")
	assertLabel(t, nodes, "Processor.process")
	assertLabel(t, nodes, "Processor.format")
	assertEdgeKind(t, edges, "IMPORTS")
}

func TestCountLineBraces(t *testing.T) {
	cases := []struct {
		line                string
		inBlock             bool
		wantOpen, wantClose int
		wantStill           bool
	}{
		{`if (x) {`, false, 1, 0, false},
		{`}`, false, 0, 1, false},
		{`foo(); // } { in a line comment`, false, 0, 0, false},
		{`a /* } */ b {`, false, 1, 0, false}, // brace in block comment ignored; real { counted
		{`x /* opens a block`, false, 0, 0, true},
		{`} still comment } `, true, 0, 0, true},  // wholly inside a block — braces ignored
		{`done */ {`, true, 1, 0, false},          // block closes, then a real {
		{`"{ not a brace }"`, false, 0, 0, false}, // string literal
	}
	for _, c := range cases {
		o, cl, still := countLineBraces(c.line, c.inBlock)
		if o != c.wantOpen || cl != c.wantClose || still != c.wantStill {
			t.Errorf("countLineBraces(%q, %v) = (%d,%d,%v), want (%d,%d,%v)",
				c.line, c.inBlock, o, cl, still, c.wantOpen, c.wantClose, c.wantStill)
		}
	}
}

// TestBlockCommentsDoNotCorruptScope verifies a stray brace inside a /* */ block
// comment does not prematurely pop the enclosing class/impl scope, which used to
// mis-attribute (or drop) methods declared after the comment.
func TestBlockCommentsDoNotCorruptScope(t *testing.T) {
	t.Run("javascript", func(t *testing.T) {
		src := `class Widget {
    first() {
        return 1;
    }
    /* a comment with stray braces } and {
       spanning lines } { still a comment */
    second() {
        return 2;
    }
}
`
		f := writeTempFile(t, "widget.js", src)
		nodes, _, err := (&JavaScriptExtractor{}).ExtractFile(context.Background(), f)
		if err != nil {
			t.Fatalf("ExtractFile: %v", err)
		}
		assertLabel(t, nodes, "Widget.first")
		assertLabel(t, nodes, "Widget.second")
	})

	t.Run("java", func(t *testing.T) {
		src := `public class Box {
    public int a() { return 1; }
    /* stray } and { inside
       a block comment } */
    public int b() { return 2; }
}
`
		f := writeTempFile(t, "Box.java", src)
		nodes, _, err := (&JavaExtractor{}).ExtractFile(context.Background(), f)
		if err != nil {
			t.Fatalf("ExtractFile: %v", err)
		}
		assertLabel(t, nodes, "Box.a")
		assertLabel(t, nodes, "Box.b")
	})

	t.Run("rust", func(t *testing.T) {
		src := `struct S;

impl S {
    fn a(&self) -> i32 { 1 }
    /* stray } and { in a
       block comment } { */
    fn b(&self) -> i32 { 2 }
}
`
		f := writeTempFile(t, "s.rs", src)
		nodes, _, err := (&RustExtractor{}).ExtractFile(context.Background(), f)
		if err != nil {
			t.Fatalf("ExtractFile: %v", err)
		}
		assertLabel(t, nodes, "S.a")
		assertLabel(t, nodes, "S.b")
	})
}

// TestAmbiguousBareCallsAreSkipped verifies that when two classes define a
// method with the same bare name, a bare-name call is not misdirected to the
// wrong class's method — the extractor skips the ambiguous edge entirely.
func TestAmbiguousBareCallsAreSkipped(t *testing.T) {
	src := `class A {
    parse() {
        return 1;
    }
    run() {
        return parse();
    }
}

class B {
    parse() {
        return 2;
    }
}
`
	f := writeTempFile(t, "amb.js", src)
	nodes, edges, err := (&JavaScriptExtractor{}).ExtractFile(context.Background(), f)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	id := map[string]string{}
	for _, n := range nodes {
		id[n.Label] = n.ID
	}
	// `parse` is defined by both A and B, so it is ambiguous; run()'s call to
	// parse() must not produce a CALLS edge to EITHER method.
	for _, e := range edges {
		if e.Kind == model.EdgeCalls && (e.Dst == id["A.parse"] || e.Dst == id["B.parse"]) {
			t.Errorf("emitted an ambiguous CALLS edge to %q", e.Dst)
		}
	}
}

func TestForFile(t *testing.T) {
	cases := []struct{ path, wantNil string }{
		{"main.go", ""},
		{"app.py", ""},
		{"index.js", ""},
		{"app.ts", ""},
		{"lib.rs", ""},
		{"Main.java", ""},
		{"README.md", ""},
		{"diagram.png", ""},
		{"data.csv", "nil"},
	}
	for _, tc := range cases {
		ex := ForFile(tc.path)
		if tc.wantNil == "nil" && ex != nil {
			t.Errorf("ForFile(%q) = %T, want nil", tc.path, ex)
		}
		if tc.wantNil == "" && ex == nil {
			t.Errorf("ForFile(%q) = nil, want extractor", tc.path)
		}
	}
}

// --- helpers ---

func writeTempFile(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := dir + "/" + name
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func logNodesEdges(t *testing.T, nodes []model.Node, edges []model.Edge) {
	t.Helper()
	t.Logf("nodes: %d, edges: %d", len(nodes), len(edges))
	for _, n := range nodes {
		t.Logf("  node: kind=%-8s label=%q", n.Kind, n.Label)
	}
	for _, e := range edges {
		t.Logf("  edge: kind=%-12s src=%.8s dst=%.8s", e.Kind, e.Src, e.Dst)
	}
}

func assertLabel(t *testing.T, nodes []model.Node, want string) {
	t.Helper()
	for _, n := range nodes {
		if n.Label == want {
			return
		}
	}
	t.Errorf("no node with label %q", want)
}

func assertEdgeKind(t *testing.T, edges []model.Edge, kind string) {
	t.Helper()
	for _, e := range edges {
		if string(e.Kind) == kind {
			return
		}
	}
	t.Errorf("no edge with kind %q", kind)
}
