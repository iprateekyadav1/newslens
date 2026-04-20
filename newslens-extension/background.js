/**
 * NewsLens Service Worker
 * Keeps the extension alive and handles tab navigation events
 * to clear stale cached results when the user navigates away.
 */

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Clear cached result when the user navigates to a new URL
  if (changeInfo.status === 'loading' && changeInfo.url) {
    chrome.storage.local.get(['lastUrl'], ({ lastUrl }) => {
      if (lastUrl && lastUrl !== changeInfo.url) {
        chrome.storage.local.remove(['lastResult', 'lastUrl']);
      }
    });
  }
});
