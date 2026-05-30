// Package jobs provides a small in-process, debouncing, coalescing worker
// queue keyed by string. It is the background job runner the live-sync engine
// (Phase 3) uses for re-extraction work.
//
// Design note — why not asynq: the plan's Phase 3 lists asynq for background
// re-extraction, but asynq requires a Redis server. A network service
// contradicts chiton's single-binary, on-device, zero-runtime-dependency
// principle — the same reasoning that put pure-Go BadgerDB ahead of CGo Kuzu
// and a pure-Go flat vector index ahead of sqlite-vec. So the v1 job runner is
// this pure-Go Queue; an asynq/Redis-backed distributed runner remains a
// pluggable future backend, exactly like the Kuzu store stub. The Queue's
// surface (Enqueue / Drain / Stop) is deliberately small so swapping in a
// distributed backend later is a contained change.
//
// Semantics:
//   - Keys are coalesced: enqueuing the same key repeatedly within the debounce
//     window collapses into a single run. Editors that write a file several
//     times in quick succession therefore trigger one re-extraction, not many.
//   - A key already running is never run concurrently with itself. If it is
//     re-enqueued while in flight, it is scheduled to run once more after the
//     current run finishes (so the final on-disk state is always processed).
//   - Distinct keys run concurrently up to Workers.
//   - Drain() processes all outstanding (debounced + queued) work before
//     returning; Stop() abandons it. Both then wait for in-flight handlers.
package jobs

import (
	"context"
	"sync"
	"time"
)

// Handler processes one coalesced key. It is called from a worker goroutine;
// distinct keys may run concurrently, so a Handler must be safe for concurrent
// use across different keys (the queue guarantees a single key never runs
// concurrently with itself).
type Handler func(ctx context.Context, key string) error

// Options configures a Queue. Zero values fall back to sane defaults.
type Options struct {
	// Workers is the maximum number of keys processed concurrently.
	// Default: 4.
	Workers int
	// Debounce is how long to wait after the last Enqueue of a key before
	// running it. Default: 250ms.
	Debounce time.Duration
	// OnError, if set, is called when a Handler returns a non-nil error.
	OnError func(key string, err error)
	// OnDone, if set, is called after a key's Handler returns (err may be nil).
	// Useful for accumulating "touched" state for a later cluster pass.
	OnDone func(key string, err error)
}

const (
	defaultWorkers  = 4
	defaultDebounce = 250 * time.Millisecond
	drainPoll       = 2 * time.Millisecond
)

// Queue is a debouncing, coalescing, bounded worker queue.
type Queue struct {
	handler  Handler
	debounce time.Duration
	onError  func(string, error)
	onDone   func(string, error)

	// ctx is the queue's own lifetime context, created in New so Enqueue is
	// safe even before Start (no nil-context panic). It is cancelled only by
	// Stop/Drain (or, via Start, by an external parent context).
	ctx    context.Context
	cancel context.CancelFunc

	ready    chan string   // keys whose debounce elapsed, awaiting a worker
	workers  chan struct{} // semaphore bounding concurrent handlers
	draining chan struct{} // closed by Drain to switch dispatch into drain mode
	wg       sync.WaitGroup

	mu      sync.Mutex
	timers  map[string]*time.Timer // keys waiting out their debounce window
	running map[string]bool        // keys currently executing
	requeue map[string]bool        // keys re-enqueued while running
	stopped bool
}

// New creates a Queue. Call Start to begin processing.
func New(handler Handler, opts Options) *Queue {
	if opts.Workers <= 0 {
		opts.Workers = defaultWorkers
	}
	if opts.Debounce <= 0 {
		opts.Debounce = defaultDebounce
	}
	q := &Queue{
		handler:  handler,
		debounce: opts.Debounce,
		onError:  opts.OnError,
		onDone:   opts.OnDone,
		ready:    make(chan string, 1024),
		workers:  make(chan struct{}, opts.Workers),
		draining: make(chan struct{}),
		timers:   map[string]*time.Timer{},
		running:  map[string]bool{},
		requeue:  map[string]bool{},
	}
	q.ctx, q.cancel = context.WithCancel(context.Background())
	return q
}

// Start launches the dispatch loop. If the parent ctx is cancelled the queue is
// hard-stopped (pending and queued work dropped), equivalent to Stop. Handlers
// receive the queue's own context, which stays live through a graceful Drain.
func (q *Queue) Start(ctx context.Context) {
	q.wg.Add(1)
	go q.dispatch()
	// Bridge an external cancel to a hard stop. If ctx has no cancel (e.g.
	// context.Background), ctx.Done() is nil and this goroutine simply waits on
	// the queue's own context, exiting cleanly when Stop/Drain cancels it.
	go func() {
		select {
		case <-ctx.Done():
			q.cancel()
		case <-q.ctx.Done():
		}
	}()
}

// Enqueue schedules key to run after the debounce window. Repeated calls within
// the window coalesce. Safe for concurrent use, including before Start.
func (q *Queue) Enqueue(key string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.stopped {
		return
	}
	if q.running[key] {
		// Already running → run exactly once more after it finishes.
		q.requeue[key] = true
		return
	}
	if t, ok := q.timers[key]; ok {
		if t.Stop() {
			// Cancelled before it fired: release its Add and re-arm fresh below,
			// extending the debounce window.
			q.wg.Done()
			delete(q.timers, key)
		} else {
			// Already fired/firing: its callback will dispatch the key. Arming a
			// new timer here would double-dispatch (and double-Done the
			// WaitGroup). The handler reads current on-disk state, and the
			// running/requeue bookkeeping coalesces any further change.
			return
		}
	}
	q.armLocked(key)
}

// armLocked starts a debounce timer for key. Caller must hold q.mu. Each arm is
// balanced by exactly one wg.Done: either the callback's defer, or a Stop()
// that returns true in Enqueue/Stop/Drain.
func (q *Queue) armLocked(key string) {
	q.wg.Add(1)
	q.timers[key] = time.AfterFunc(q.debounce, func() {
		defer q.wg.Done()
		// Send to ready BEFORE removing the timer entry, so a Drain idle-check
		// that observes the entry gone is guaranteed to also see the key queued
		// (no lost work in the fire-during-drain window).
		select {
		case q.ready <- key:
		case <-q.ctx.Done():
		}
		q.mu.Lock()
		delete(q.timers, key)
		q.mu.Unlock()
	})
}

// dispatch pulls ready keys and runs them under the worker semaphore.
func (q *Queue) dispatch() {
	defer q.wg.Done()
	for {
		select {
		case <-q.ctx.Done():
			return // hard stop: abandon queued work
		case <-q.draining:
			q.drainLoop()
			return
		case key := <-q.ready:
			q.launch(key)
		}
	}
}

// drainLoop processes every outstanding key (queued and not-yet-fired timers
// that Drain hands over) until the pipeline is fully quiescent, then returns.
func (q *Queue) drainLoop() {
	for {
		select {
		case key := <-q.ready:
			q.launch(key)
			continue
		default:
		}
		q.mu.Lock()
		idle := len(q.timers) == 0 && len(q.running) == 0
		q.mu.Unlock()
		if idle && len(q.ready) == 0 {
			return
		}
		select {
		case <-q.ctx.Done():
			return
		case <-time.After(drainPoll):
		}
	}
}

// launch acquires a worker slot and runs key in a fresh goroutine.
func (q *Queue) launch(key string) {
	select {
	case q.workers <- struct{}{}:
	case <-q.ctx.Done():
		return
	}
	q.mu.Lock()
	q.running[key] = true
	q.mu.Unlock()
	q.wg.Add(1)
	go func(k string) {
		defer q.wg.Done()
		defer func() { <-q.workers }()
		q.run(k)
	}(key)
}

// run executes the handler for one key and re-enqueues it if it was touched
// while running.
func (q *Queue) run(key string) {
	err := q.handler(q.ctx, key)
	if err != nil && q.onError != nil {
		q.onError(key, err)
	}
	if q.onDone != nil {
		q.onDone(key, err)
	}

	q.mu.Lock()
	delete(q.running, key)
	again := q.requeue[key]
	delete(q.requeue, key)
	stopped := q.stopped
	q.mu.Unlock()

	if again && !stopped {
		q.Enqueue(key)
	}
}

// Drain stops accepting new work, processes everything already enqueued
// (including timers that have not yet fired), waits for all handlers to finish,
// then shuts down. Safe to call once; a second call is a no-op.
func (q *Queue) Drain() {
	q.mu.Lock()
	if q.stopped {
		q.mu.Unlock()
		return
	}
	q.stopped = true
	var pending []string
	for key, t := range q.timers {
		if t.Stop() {
			q.wg.Done() // balance the arming Add
			delete(q.timers, key)
			pending = append(pending, key) // still owed a run
		}
		// Stop()==false: the callback already fired; it will queue the key to
		// ready and delete its own timer entry, so drainLoop will process it.
	}
	q.mu.Unlock()

	// Hand the cancelled-but-pending keys to ready, then switch dispatch into
	// drain mode (push first so drainLoop's idle-check never fires early).
	for _, key := range pending {
		select {
		case q.ready <- key:
		case <-q.ctx.Done():
		}
	}
	close(q.draining)

	q.wg.Wait()
	q.cancel()
}

// Stop stops accepting new work, abandons pending/queued work, cancels in-flight
// handlers' context, and waits for them to return. Safe to call once.
func (q *Queue) Stop() {
	q.mu.Lock()
	if q.stopped {
		q.mu.Unlock()
		return
	}
	q.stopped = true
	for key, t := range q.timers {
		if t.Stop() {
			q.wg.Done()
		}
		delete(q.timers, key)
	}
	q.mu.Unlock()

	q.cancel()
	q.wg.Wait()
}
