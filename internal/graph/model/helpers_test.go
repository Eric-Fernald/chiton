package model

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

func TestSlug(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"Hello, World!", "hello-world"},
		{"", "fb"},
		{"---", "fb"},
		{"Émile Zola", "mile-zola"},
		{"a__b", "a-b"},
		{"ÀB", "b"},
		{"\u212Aelvin", "kelvin"}, // Kelvin sign lowercases to ASCII k
		{"İstanbul", "istanbul"},
		{"Sprint 42 — Retro", "sprint-42-retro"},
		{"  x  ", "x"},
		{"日本語", "fb"},
		{"a\xffb", "a-b"},
		{"Q3/Q4 planning (v2)", "q3-q4-planning-v2"},
	} {
		if got := Slug(tc.in, "fb"); got != tc.want {
			t.Errorf("Slug(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestContentSHA(t *testing.T) {
	for in, want := range map[string]string{
		"":    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		"abc": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
	} {
		if got := ContentSHA([]byte(in)); got != want {
			t.Errorf("ContentSHA(%q) = %s, want %s", in, got, want)
		}
	}
}

func TestParseEdgeKinds(t *testing.T) {
	if got := ParseEdgeKinds(nil); got != nil {
		t.Errorf("ParseEdgeKinds(nil) = %v, want nil", got)
	}
	if got := ParseEdgeKinds([]string{}); got != nil {
		t.Errorf("ParseEdgeKinds(empty) = %v, want nil", got)
	}
	got := ParseEdgeKinds([]string{"calls", "Imports", "DEFINED_in", ""})
	if want := []EdgeKind{EdgeCalls, EdgeImports, EdgeDefinedIn, ""}; !slices.Equal(got, want) {
		t.Errorf("ParseEdgeKinds = %v, want %v", got, want)
	}
}

func TestFileURI(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]string{
		"a/b.go":   "file://" + filepath.ToSlash(filepath.Join(wd, "a", "b.go")),
		"./x/../y": "file://" + filepath.ToSlash(filepath.Join(wd, "y")),
	}
	if runtime.GOOS == "windows" {
		cases[`C:\src\app\main.go`] = "file://C:/src/app/main.go"
		cases[`C:\src\..\x.go`] = "file://C:/x.go"
	} else {
		cases["/src/app/main.go"] = "file:///src/app/main.go"
		cases["/src/../x.go"] = "file:///x.go"
	}
	for in, want := range cases {
		if got := FileURI(in); got != want {
			t.Errorf("FileURI(%q) = %q, want %q", in, got, want)
		}
		// Round trip back to the cleaned absolute path.
		abs, _ := filepath.Abs(in)
		if p, ok := PathFromFileURI(FileURI(in)); !ok || p != abs {
			t.Errorf("PathFromFileURI(FileURI(%q)) = %q, %v; want %q", in, p, ok, abs)
		}
	}
}

func TestPathFromFileURI(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
		ok   bool
	}{
		{"file:///a/b.go", filepath.FromSlash("/a/b.go"), true},
		{"file://C:/x/y.go", filepath.FromSlash("C:/x/y.go"), true},
		{"file://", "", true},
		{"FILE:///a", "", false}, // the scheme match is case-sensitive, like the call sites it replaces
		{"http://x/y", "", false},
		{"meeting://standup", "", false},
		{"", "", false},
	} {
		if got, ok := PathFromFileURI(tc.in); got != tc.want || ok != tc.ok {
			t.Errorf("PathFromFileURI(%q) = %q, %v; want %q, %v", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}
