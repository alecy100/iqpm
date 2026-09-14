import { useCallback, useEffect, useRef, useState } from "react";

/** Polls `fn` every `intervalMs`, pausing while `paused` is true. Returns [data, error, refresh]. */
export function usePolling(fn, deps, intervalMs, paused = false) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const refresh = useCallback(async () => {
    try {
      const result = await fnRef.current();
      setData(result);
      setError(null);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let alive = true;
    let timer;
    const tick = async () => {
      try {
        const result = await fnRef.current();
        if (alive) {
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (alive) setError(err.message);
      }
      if (alive) timer = setTimeout(tick, intervalMs);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, intervalMs, paused]);

  return [data, error, refresh];
}

const MIN_PANE_PX = 300;

/** Horizontal IDE-style resizable panes: drag the gutter between two panes; closing one gives its
 * space to its neighbour instead of shrinking everything. */
export function useSplitPanes(initialIds) {
  const [order, setOrder] = useState(initialIds);
  const [fractions, setFractions] = useState(() => equalFractions(initialIds.length));
  const containerRef = useRef(null);
  const drag = useRef(null);

  const addPane = useCallback((id) => {
    setOrder((prev) => [...prev, id]);
    setFractions((prev) => {
      const shrink = 0.32;
      const rest = prev.map((value) => value * (1 - shrink));
      return [...rest, shrink];
    });
  }, []);

  const removePane = useCallback((id) => {
    setOrder((prevOrder) => {
      const index = prevOrder.indexOf(id);
      if (index === -1) return prevOrder;
      setFractions((prevFractions) => {
        if (prevFractions.length <= 1) return prevFractions;
        const removed = prevFractions[index];
        const rest = prevFractions.filter((_, i) => i !== index);
        const neighbourIndex = index === 0 ? 0 : index - 1;
        rest[neighbourIndex] += removed;
        return rest;
      });
      return prevOrder.filter((_, i) => i !== index);
    });
  }, []);

  const startDrag = useCallback((gutterIndex) => (event) => {
    event.preventDefault();
    const container = containerRef.current;
    if (!container) return;
    drag.current = {
      gutterIndex,
      startX: event.clientX,
      startFractions: fractions,
      containerWidth: container.getBoundingClientRect().width,
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopDrag);
    document.body.style.cursor = "col-resize";
  }, [fractions]);

  const handlePointerMove = (event) => {
    const state = drag.current;
    if (!state) return;
    const { gutterIndex, startX, startFractions, containerWidth } = state;
    const deltaFraction = (event.clientX - startX) / containerWidth;
    const left = gutterIndex;
    const right = gutterIndex + 1;
    const minFraction = MIN_PANE_PX / containerWidth;
    const pairTotal = startFractions[left] + startFractions[right];
    let newLeft = startFractions[left] + deltaFraction;
    newLeft = Math.max(minFraction, Math.min(pairTotal - minFraction, newLeft));
    setFractions((prev) => {
      const next = [...prev];
      next[left] = newLeft;
      next[right] = pairTotal - newLeft;
      return next;
    });
  };

  const stopDrag = () => {
    drag.current = null;
    window.removeEventListener("pointermove", handlePointerMove);
    window.removeEventListener("pointerup", stopDrag);
    document.body.style.cursor = "";
  };

  useEffect(() => () => stopDrag(), []);

  return { order, fractions, containerRef, addPane, removePane, startDrag };
}

function equalFractions(count) {
  return Array.from({ length: count }, () => 1 / count);
}
