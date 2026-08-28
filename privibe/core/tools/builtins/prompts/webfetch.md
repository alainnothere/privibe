Fetches content from a specified URL and converts HTML to markdown for readability.
Use this tool when you need to retrieve and analyze web content.

- Prefer a more specialized tool over `web_fetch` when one is available.
- URLs must be valid.
- Read-only: does not modify any files.
- To search the web when no `web_search` tool is available, fetch
  `https://html.duckduckgo.com/html/?q=your+query` — a plain-HTML result page
  that converts cleanly to markdown.
- Non-text responses (PDFs, images, archives, other binaries) are not returned
  into the conversation: the file is saved to disk and the result gives its
  path, so you can inspect or convert it with local tools.
