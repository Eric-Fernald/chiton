package bearer

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
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
