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

