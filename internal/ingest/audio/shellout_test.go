package audio

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestShellTranscribeParsesVTT(t *testing.T) {
	audio := filepath.Join(t.TempDir(), "standup.mp3")
	if err := os.WriteFile(audio, []byte("fake audio"), 0o644); err != nil {
		t.Fatal(err)
	}

	st := &ShellTranscriber{
		cfg: Config{Binary: "whisper", Model: "base"},
		// Fake whisper: write a VTT into the output dir (the last arg after
		// --output_dir for the OpenAI form).
		run: func(_ context.Context, _, _ string, args []string) error {
			outDir := argAfter(args, "--output_dir")
			if outDir == "" {
				t.Fatalf("no --output_dir in args: %v", args)
			}
			vtt := "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello team.\n\n" +
				"00:00:02.000 --> 00:00:04.000\nLet's begin.\n"
			return os.WriteFile(filepath.Join(outDir, "standup.vtt"), []byte(vtt), 0o644)
		},
	}

	tr, err := st.Transcribe(context.Background(), audio)
	if err != nil {
		t.Fatal(err)
	}
	if tr.Title != "standup" {
		t.Errorf("title=%q want standup", tr.Title)
	}
	if len(tr.Turns) != 2 {
		t.Fatalf("turns=%d want 2: %+v", len(tr.Turns), tr.Turns)
	}
	if !strings.Contains(tr.Turns[0].Text, "Hello team") {
		t.Errorf("turn0=%q", tr.Turns[0].Text)
	}
}

func TestBuildArgsWhisperCpp(t *testing.T) {
	st := &ShellTranscriber{cfg: Config{Model: "/models/ggml-base.bin", Language: "en"}}
	args, err := st.buildArgs("whisper-cli", "rec.wav", "/tmp/out")
	if err != nil {
		t.Fatal(err)
	}
	if argAfter(args, "-m") != "/models/ggml-base.bin" || argAfter(args, "-f") != "rec.wav" {
		t.Errorf("whisper.cpp args wrong: %v", args)
	}
	if argAfter(args, "-l") != "en" {
		t.Errorf("language flag missing: %v", args)
	}
}

func TestBuildArgsWhisperCppRequiresModel(t *testing.T) {
	st := &ShellTranscriber{cfg: Config{}}
	if _, err := st.buildArgs("whisper-cli", "rec.wav", "/tmp/out"); err == nil {
		t.Error("expected error: whisper.cpp needs a model path")
	}
}

func TestResolveBinaryConfigured(t *testing.T) {
	st := &ShellTranscriber{cfg: Config{Binary: "my-whisper"}}
	got, err := st.resolveBinary()
	if err != nil || got != "my-whisper" {
		t.Errorf("resolveBinary=%q,%v want my-whisper,nil", got, err)
	}
}

func argAfter(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}
