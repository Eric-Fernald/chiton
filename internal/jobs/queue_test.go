package jobs

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestCoalesce verifies repeated enqueues of one key within the debounce window
// collapse into a single handler run.
func TestCoalesce(t *testing.T) {
	var runs int32
	q := New(func(_ context.Context, _ string) error {
		atomic.AddInt32(&runs, 1)
		return nil
	}, Options{Workers: 2, Debounce: 30 * time.Millisecond})
	q.Start(context.Background())
	defer q.Stop()

	for i := 0; i < 10; i++ {
		q.Enqueue("a")
	}
	time.Sleep(200 * time.Millisecond)

	if got := atomic.LoadInt32(&runs); got != 1 {
		t.Fatalf("expected 1 coalesced run, got %d", got)
	}
}

// TestDistinctKeysRun verifies that distinct keys each run.
func TestDistinctKeysRun(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]int{}
	done := make(chan struct{}, 3)
	q := New(func(_ context.Context, key string) error {
		mu.Lock()
		seen[key]++
		mu.Unlock()
		done <- struct{}{}
		return nil
	}, Options{Workers: 3, Debounce: 10 * time.Millisecond})
	q.Start(context.Background())
	defer q.Stop()

	q.Enqueue("a")
	q.Enqueue("b")
	q.Enqueue("c")

	for i := 0; i < 3; i++ {
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for key %d", i)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	for _, k := range []string{"a", "b", "c"} {
		if seen[k] != 1 {
			t.Errorf("key %q ran %d times, want 1", k, seen[k])
		}
	}
}

// TestRequeueWhileRunning verifies a key enqueued while its handler is running
// runs exactly once more after the in-flight run completes.
func TestRequeueWhileRunning(t *testing.T) {
	var runs int32
	started := make(chan struct{})
	release := make(chan struct{})
	q := New(func(_ context.Context, _ string) error {
		n := atomic.AddInt32(&runs, 1)
		if n == 1 {
			close(started)
			<-release // hold the first run open
		}
		return nil
	}, Options{Workers: 1, Debounce: 10 * time.Millisecond})
	q.Start(context.Background())
	defer q.Stop()

	q.Enqueue("x")
	<-started      // first run is now in flight
	q.Enqueue("x") // should be remembered and re-run after release
	q.Enqueue("x") // collapses into the same pending re-run
	close(release)

	deadline := time.After(2 * time.Second)
	for atomic.LoadInt32(&runs) < 2 {
		select {
		case <-deadline:
			t.Fatalf("expected a second run, got %d", atomic.LoadInt32(&runs))
		case <-time.After(10 * time.Millisecond):
		}
	}
	time.Sleep(100 * time.Millisecond)
	if got := atomic.LoadInt32(&runs); got != 2 {
		t.Fatalf("expected exactly 2 runs, got %d", got)
	}
}

// TestConcurrency verifies up to Workers keys run at the same time.
func TestConcurrency(t *testing.T) {
	const workers = 3
	var inFlight int32
	var maxSeen int32
	barrier := make(chan struct{})
	q := New(func(_ context.Context, _ string) error {
		cur := atomic.AddInt32(&inFlight, 1)
		for {
			old := atomic.LoadInt32(&maxSeen)
			if cur <= old || atomic.CompareAndSwapInt32(&maxSeen, old, cur) {
				break
			}
		}
		<-barrier
		atomic.AddInt32(&inFlight, -1)
		return nil
	}, Options{Workers: workers, Debounce: 10 * time.Millisecond})
	q.Start(context.Background())

	for _, k := range []string{"a", "b", "c"} {
		q.Enqueue(k)
	}
	// Wait until all three are concurrently in flight (or time out).
	deadline := time.After(2 * time.Second)
	for atomic.LoadInt32(&inFlight) < workers {
		select {
		case <-deadline:
			t.Fatalf("only %d workers ran concurrently, want %d", atomic.LoadInt32(&maxSeen), workers)
		case <-time.After(5 * time.Millisecond):
		}
	}
	close(barrier)
	q.Stop()
	if maxSeen < workers {
		t.Fatalf("max concurrency %d < %d", maxSeen, workers)
	}
}

// TestStopBeforeFire cancels the debounce window cleanly with no panics/leaks.
func TestStopBeforeFire(t *testing.T) {
	var runs int32
	q := New(func(_ context.Context, _ string) error {
		atomic.AddInt32(&runs, 1)
		return nil
	}, Options{Workers: 1, Debounce: 500 * time.Millisecond})
	q.Start(context.Background())
	q.Enqueue("a")
	q.Stop() // before the 500ms window elapses
	if got := atomic.LoadInt32(&runs); got != 0 {
		t.Fatalf("handler ran %d times despite early Stop", got)
	}
	// Enqueue after Stop is a no-op (must not panic).
	q.Enqueue("b")
}

// TestDrainProcessesPending verifies Drain runs work whose debounce window has
// NOT yet elapsed (unlike Stop, which abandons it).
func TestDrainProcessesPending(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]bool{}
	q := New(func(_ context.Context, key string) error {
		mu.Lock()
		seen[key] = true
		mu.Unlock()
		return nil
	}, Options{Workers: 4, Debounce: 10 * time.Second}) // long debounce
	q.Start(context.Background())

	for _, k := range []string{"a", "b", "c", "d"} {
		q.Enqueue(k)
	}
	q.Drain() // must process all four despite the 10s debounce

	mu.Lock()
	defer mu.Unlock()
	for _, k := range []string{"a", "b", "c", "d"} {
		if !seen[k] {
			t.Errorf("Drain did not process pending key %q", k)
		}
	}
}

// TestDrainProcessesRequeuedWork verifies Drain processes work re-enqueued
// while a handler was in flight, instead of dropping the key's final state.
func TestDrainProcessesRequeuedWork(t *testing.T) {
	var runs int32
	firstStarted := make(chan struct{})
	proceed := make(chan struct{})
	q := New(func(_ context.Context, _ string) error {
		if atomic.AddInt32(&runs, 1) == 1 {
			close(firstStarted)
			<-proceed // hold the first run open so we can re-enqueue + drain mid-flight
		}
		return nil
	}, Options{Workers: 1, Debounce: 5 * time.Millisecond})
	q.Start(context.Background())

	q.Enqueue("x")
	<-firstStarted
	q.Enqueue("x") // re-enqueued while the first handler runs → owed a re-run

	drained := make(chan struct{})
	go func() { q.Drain(); close(drained) }()

	time.Sleep(30 * time.Millisecond) // let Drain set stopped while the handler is held
	close(proceed)                    // release the first run

	select {
	case <-drained:
	case <-time.After(2 * time.Second):
		t.Fatal("Drain did not return")
	}
	if got := atomic.LoadInt32(&runs); got < 2 {
		t.Fatalf("requeued work dropped on drain: handler ran %d times, want >= 2", got)
	}
}

// TestStressEnqueueDuringFire hammers Enqueue against timer fires on a tiny debounce, the interleaving
// that once double-fired a callback and panicked on a negative WaitGroup; not panicking is the guard.
func TestStressEnqueueDuringFire(t *testing.T) {
	var runs int32
	q := New(func(_ context.Context, _ string) error {
		atomic.AddInt32(&runs, 1)
		return nil
	}, Options{Workers: 8, Debounce: time.Millisecond})
	q.Start(context.Background())

	keys := []string{"a", "b", "c", "d", "e"}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				for _, k := range keys {
					q.Enqueue(k)
				}
			}
		}()
	}
	time.Sleep(300 * time.Millisecond)
	close(stop)
	wg.Wait()
	q.Drain()

	if atomic.LoadInt32(&runs) == 0 {
		t.Fatal("expected handler to have run at least once")
	}
}

// TestNoConcurrentSelfRun guards that a key never runs concurrently with itself; each run is held open
// to widen the race that lands a key in ready twice. Run with -race for full coverage.
func TestNoConcurrentSelfRun(t *testing.T) {
	var mu sync.Mutex
	inFlight := map[string]int{}
	var violated int32
	q := New(func(_ context.Context, key string) error {
		mu.Lock()
		inFlight[key]++
		if inFlight[key] > 1 {
			atomic.StoreInt32(&violated, 1)
		}
		mu.Unlock()

		time.Sleep(time.Millisecond) // hold the run open to widen any overlap

		mu.Lock()
		inFlight[key]--
		mu.Unlock()
		return nil
	}, Options{Workers: 8, Debounce: time.Millisecond})
	q.Start(context.Background())

	keys := []string{"a", "b", "c", "d", "e"}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				for _, k := range keys {
					q.Enqueue(k)
				}
			}
		}()
	}
	time.Sleep(300 * time.Millisecond)
	close(stop)
	wg.Wait()
	q.Drain()

	if atomic.LoadInt32(&violated) != 0 {
		t.Fatal("a key ran concurrently with itself (single-key-serialization invariant violated)")
	}
}

// TestOnDone fires after each handler run.
func TestOnDone(t *testing.T) {
	var doneCount int32
	got := make(chan struct{}, 1)
	q := New(func(_ context.Context, _ string) error { return nil }, Options{
		Workers:  1,
		Debounce: 10 * time.Millisecond,
		OnDone: func(string, error) {
			atomic.AddInt32(&doneCount, 1)
			select {
			case got <- struct{}{}:
			default:
			}
		},
	})
	q.Start(context.Background())
	defer q.Stop()
	q.Enqueue("a")
	select {
	case <-got:
	case <-time.After(2 * time.Second):
		t.Fatal("OnDone never fired")
	}
}
