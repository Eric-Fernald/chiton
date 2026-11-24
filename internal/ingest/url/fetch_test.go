package url

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// fakeYtdlpEnv selects how the test binary behaves when fetchYouTube runs it in place of yt-dlp.
const (
	fakeYtdlpEnv     = "CHITON_TEST_FAKE_YTDLP"      // ok | fail | hang
	fakeYtdlpArgsEnv = "CHITON_TEST_FAKE_YTDLP_ARGS" // file the fake writes its arguments to, one per line
)

// TestMain turns the test binary into a stand-in yt-dlp when fakeYtdlpEnv is set, so no test needs the real one.
func TestMain(m *testing.M) {
	if mode := os.Getenv(fakeYtdlpEnv); mode != "" {
		os.Exit(fakeYtdlp(mode, os.Args[1:]))
	}
	os.Exit(m.Run())
}

func fakeYtdlp(mode string, args []string) int {
	if path := os.Getenv(fakeYtdlpArgsEnv); path != "" {
		if err := os.WriteFile(path, []byte(strings.Join(args, "\n")), 0o600); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
	}
	switch mode {
	case "fail":
		fmt.Fprint(os.Stderr, strings.Repeat("WARNING: retrying the player API\n", 400))
		fmt.Fprintln(os.Stderr, "ERROR: [youtube] abc: Sign in to confirm your age")
		return 1
	case "hang":
		time.Sleep(5 * time.Second)
		return 0
	}
	i := slices.Index(args, "-o")
	if i < 0 || i+1 >= len(args) {
		fmt.Fprintln(os.Stderr, "fake yt-dlp: no -o template")
		return 2
	}
	vtt := "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nWelcome to the show.\n"
	if err := os.WriteFile(filepath.Join(filepath.Dir(args[i+1]), "abc.en.vtt"), []byte(vtt), 0o600); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	fmt.Println("Fake Title")
	return 0
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// fakeYtdlpFetcher returns a fetcher whose yt-dlp is this test binary in the given mode, and whose HTTP client
// refuses to touch the network.
func fakeYtdlpFetcher(t *testing.T, mode string) *httpFetcher {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeYtdlpEnv, mode)
	noNetwork := roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("test reached the network instead of yt-dlp")
	})
	return &httpFetcher{client: &http.Client{Transport: noNetwork}, ytdlp: self}
}

func TestYouTubeHostWithPortGoesToYtdlp(t *testing.T) {
	h := fakeYtdlpFetcher(t, "ok")
	for _, raw := range []string{"https://www.youtube.com:443/watch?v=abc", "https://youtu.be:443/abc"} {
		fr, err := h.fetch(context.Background(), raw)
		if err != nil {
			t.Fatalf("fetch(%s): %v", raw, err)
		}
		if fr.title != "Fake Title" || fr.contentType != "text/vtt" || !strings.Contains(string(fr.body), "Welcome to the show.") {
			t.Errorf("fetch(%s) = %+v; want the yt-dlp title and subtitles", raw, fr)
		}
		if kind := inferKind(fr, raw); kind != "video" {
			t.Errorf("inferKind(%s) = %q, want video", raw, kind)
		}
	}
}

// A watch URL carrying &list= would otherwise pull subtitles for the whole playlist and ingest whichever sorts first.
func TestFetchYouTubeAsksForOneVideo(t *testing.T) {
	h := fakeYtdlpFetcher(t, "ok")
	argsFile := filepath.Join(t.TempDir(), "args")
	t.Setenv(fakeYtdlpArgsEnv, argsFile)
	const raw = "https://www.youtube.com/watch?v=abc&list=PLxyz"
	if _, err := h.fetch(context.Background(), raw); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(string(got), "\n")
	if !slices.Contains(args, "--no-playlist") {
		t.Errorf("yt-dlp ran without --no-playlist: %q", args)
	}
	if args[len(args)-1] != raw {
		t.Errorf("last yt-dlp argument = %q, want the URL", args[len(args)-1])
	}
}

func TestFetchYouTubeErrorCarriesStderrTail(t *testing.T) {
	h := fakeYtdlpFetcher(t, "fail")
	_, err := h.fetch(context.Background(), "https://www.youtube.com/watch?v=abc")
	if err == nil {
		t.Fatal("fetch succeeded although yt-dlp exited 1")
	}
	if !strings.Contains(err.Error(), "Sign in to confirm your age") {
		t.Errorf("error does not say why yt-dlp failed: %v", err)
	}
	if len(err.Error()) > 2*maxStderrTail {
		t.Errorf("error is %d bytes; want yt-dlp's stderr capped near %d", len(err.Error()), maxStderrTail)
	}
}

func TestFetchYouTubeTimesOut(t *testing.T) {
	h := fakeYtdlpFetcher(t, "hang")
	h.ytdlpWait = 300 * time.Millisecond
	_, err := h.fetch(context.Background(), "https://www.youtube.com/watch?v=abc")
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("fetch of a hanging yt-dlp = %v; want a timeout error", err)
	}
}
