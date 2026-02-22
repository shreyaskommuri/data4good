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

  const refetch = useCallback(() => {
    if (!enabled) return;

    // Only show loading state on initial load (when we don't have data yet)
    // For subsequent updates, update data in background without showing loading
    const isInitialLoad = !hasInitialData.current;

    if (isInitialLoad) {
      setLoading(true);
    }
    setError(null);

    fetcher()
      .then((newData) => {
        setData(newData);
        hasInitialData.current = true;
      })
      .catch(setError)
      .finally(() => {
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

  useEffect(() => {
    if (targetValue === displayValue) return;

    startValueRef.current = displayValue;
    startTimeRef.current = null;

    const animate = (currentTime) => {
      if (!startTimeRef.current) {
        startTimeRef.current = currentTime;
      }

      const elapsed = currentTime - startTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);

      // Easing function (ease-out)
      const easeOut = 1 - Math.pow(1 - progress, 3);

      const current = startValueRef.current + (targetValue - startValueRef.current) * easeOut;
      setDisplayValue(current);

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayValue(targetValue);
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [targetValue, duration, displayValue]);

  return displayValue;
}
