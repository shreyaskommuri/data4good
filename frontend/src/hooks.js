import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * useApi — fetch data reactively.
 * @param {Function} fetcher  Async function returning the data.
 * @param {Array}    deps     Re-fetch when these values change.
 * @param {Object}   options
 * @param {boolean}  options.enabled  When false, skips the fetch (default: true).
 *                                    Useful for deferring secondary calls until
 *                                    critical data has landed.
 */
export function useApi(fetcher, deps = [], { enabled = true } = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState(null);
  const hasInitialData = useRef(false);
  const requestIdRef = useRef(0);

  const refetch = useCallback(() => {
    if (!enabled) return;

    const isInitialLoad = !hasInitialData.current;
    const thisRequestId = ++requestIdRef.current;

    if (isInitialLoad) {
      setLoading(true);
    }
    setError(null);

    fetcher()
      .then((newData) => {
        if (thisRequestId !== requestIdRef.current) return;
        setData(newData);
        hasInitialData.current = true;
      })
      .catch((err) => {
        if (thisRequestId !== requestIdRef.current) return;
        setError(err);
      })
      .finally(() => {
        if (thisRequestId !== requestIdRef.current) return;
        if (isInitialLoad) {
          setLoading(false);
        }
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled]);

  useEffect(() => { refetch(); }, [refetch]);

  return { data, loading, error, refetch };
}

// Smooth number animation hook
export function useAnimatedNumber(targetValue, duration = 500) {
  const [displayValue, setDisplayValue] = useState(targetValue);
  const animationFrameRef = useRef(null);
  const startValueRef = useRef(targetValue);
  const startTimeRef = useRef(null);
  const targetRef = useRef(targetValue);

  useEffect(() => {
    if (targetValue === targetRef.current && targetValue === displayValue) return;

    targetRef.current = targetValue;
    startValueRef.current = displayValue;
    startTimeRef.current = null;

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const animate = (currentTime) => {
      if (!startTimeRef.current) {
        startTimeRef.current = currentTime;
      }

      const elapsed = currentTime - startTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const easeInOut = progress < 0.5
        ? 16 * progress ** 5
        : 1 - (-2 * progress + 2) ** 5 / 2;

      const current = startValueRef.current + (targetRef.current - startValueRef.current) * easeInOut;
      setDisplayValue(current);

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayValue(targetRef.current);
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  // Only re-run when target changes, NOT when displayValue changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetValue, duration]);

  return displayValue;
}
