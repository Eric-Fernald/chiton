package bearer

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"
	"time"
)

func TestGetSendsBearerAndReturnsStatusAndBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.Header.Get("Authorization") != "Bearer tok" {
			t.Errorf("got %s with Authorization %q", r.Method, r.Header.Get("Authorization"))
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(" busy \n"))
	}))
	defer srv.Close()

	status, body, err := Get(context.Background(), srv.Client(), srv.URL+"/x?q=1", "tok")
	if err != nil {
		t.Fatal(err)
	}
	if status != http.StatusServiceUnavailable || string(body) != " busy \n" {
		t.Fatalf("got status %d body %q", status, body)
	}
}

func TestGetBadURLErrors(t *testing.T) {
	if _, _, err := Get(context.Background(), http.DefaultClient, "://bad", "tok"); err == nil {
		t.Fatal("expected an error for an unparseable URL")
	}
}

func stubSleep(t *testing.T, waits *[]time.Duration, err error) {
	t.Helper()
	prev := sleep
	sleep = func(_ context.Context, d time.Duration) error {
		if waits != nil {
			*waits = append(*waits, d)
		}
		return err
	}
	t.Cleanup(func() { sleep = prev })
}

func TestGetRetriesTooManyRequests(t *testing.T) {
	var waits []time.Duration
	stubSleep(t, &waits, nil)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	status, body, err := Get(context.Background(), srv.Client(), srv.URL, "tok")
	if err != nil || status != http.StatusOK || string(body) != "ok" {
		t.Fatalf("got %d %q %v, want 200 ok after one retry", status, body, err)
	}
	if !slices.Equal(waits, []time.Duration{7 * time.Second}) {
		t.Errorf("waits = %v, want [7s] from Retry-After", waits)
	}
}

func TestGetGivesUpAfterMaxRetries(t *testing.T) {
	var waits []time.Duration
	stubSleep(t, &waits, nil)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	status, _, err := Get(context.Background(), srv.Client(), srv.URL, "tok")
	if err != nil || status != http.StatusTooManyRequests || calls.Load() != 4 {
		t.Fatalf("status=%d err=%v calls=%d, want 429 after 1+3 attempts", status, err, calls.Load())
	}
	if !slices.Equal(waits, []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}) {
		t.Errorf("waits = %v, want backoff 1s 2s 4s without Retry-After", waits)
	}
}

func TestGetStopsWaitingWhenCancelled(t *testing.T) {
	stubSleep(t, nil, context.Canceled)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	if _, _, err := Get(context.Background(), srv.Client(), srv.URL, "tok"); err != context.Canceled {
		t.Fatalf("err = %v, want context.Canceled from the interrupted wait", err)
	}
}

func TestRetryDelay(t *testing.T) {
	for _, tc := range []struct {
		header  string
		attempt int
		want    time.Duration
	}{
		{"7", 0, 7 * time.Second},
		{"", 2, 4 * time.Second},
		{"999999", 0, maxRetryWait},
		{"junk", 0, time.Second},
	} {
		if got := retryDelay(tc.header, tc.attempt); got != tc.want {
			t.Errorf("retryDelay(%q, %d) = %v, want %v", tc.header, tc.attempt, got, tc.want)
		}
	}
}
