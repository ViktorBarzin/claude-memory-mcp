/**
 * OpenClaw Memory API Plugin
 *
 * Provides memory_store, memory_recall, memory_list, memory_delete tools
 * backed by the shared PostgreSQL memory API (claude-memory service).
 */

const PLUGIN_ID = "memory-api";

// ADR-0007: hard bound on Memory content, in UNICODE CHARACTERS (code points, not
// bytes and not UTF-16 units). The API rejects oversize stores with HTTP 422; the
// plugin pre-validates with the same split-into-linked-memories guidance.
const MAX_CONTENT_CHARS = 1400;

function contentBoundMessage(nChars) {
  return (
    `Content is ${nChars} characters — over the 1,400-character Memory bound ` +
    `(ADR-0007: a recalled Memory must arrive whole). Split the knowledge into ` +
    `several self-contained memories — each understandable on its own, never ` +
    `'part N of M' fragments — and link them (part-of → hub, see-also for ` +
    `related) instead of storing one oversized entry.`
  );
}

const memoryApiPlugin = {
  id: PLUGIN_ID,
  name: "Memory (API)",
  description: "PostgreSQL-backed shared memory via claude-memory API",
  kind: "memory",
  configSchema: {
    type: "object",
    additionalProperties: false,
    properties: {},
  },
  register(api) {
    const apiUrl =
      process.env.MEMORY_API_URL || "http://claude-memory.claude-memory.svc.cluster.local";
    const apiKey = process.env.MEMORY_API_KEY || "";

    async function apiRequest(method, path, body) {
      const url = `${apiUrl}${path}`;
      const headers = {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      };
      const options = { method, headers };
      if (body) {
        options.body = JSON.stringify(body);
      }
      const resp = await fetch(url, options);
      if (!resp.ok) {
        const text = await resp.text();
        // FastAPI errors carry the human message in a JSON "detail" field (e.g. the
        // 422 oversize-content guidance) — surface that instead of the raw envelope.
        let detail = "";
        try {
          const parsed = JSON.parse(text);
          if (typeof parsed?.detail === "string") detail = parsed.detail;
        } catch {
          // not JSON — fall through to the raw body
        }
        const err = new Error(`API error ${resp.status}: ${detail || text}`);
        err.status = resp.status;
        throw err;
      }
      return resp.json();
    }

    // memory_store
    api.registerTool({
      name: "memory_store",
      label: "Memory Store",
      description:
        "Store a fact or memory in persistent shared storage. Use to remember preferences, projects, decisions, or people.",
      parameters: {
        type: "object",
        properties: {
          content: {
            type: "string",
            description:
              "The fact or memory to store (max 1,400 characters — split longer knowledge into several self-contained memories)",
          },
          category: {
            type: "string",
            enum: ["facts", "preferences", "projects", "people", "decisions"],
            description: "Category for organizing the memory",
          },
          tags: { type: "string", description: "Comma-separated tags" },
          importance: {
            type: "number",
            description: "Importance 0.0-1.0",
            minimum: 0.0,
            maximum: 1.0,
          },
          expanded_keywords: {
            type: "string",
            description:
              "REQUIRED. Space-separated semantically related search terms (MINIMUM 5 words).",
          },
        },
        required: ["content", "expanded_keywords"],
      },
      async execute(_toolCallId, params) {
        // Count unicode code points (Array.from), not UTF-16 units (.length).
        const nChars = Array.from(params.content ?? "").length;
        if (nChars > MAX_CONTENT_CHARS) {
          return {
            content: [{ type: "text", text: `Cannot store: ${contentBoundMessage(nChars)}` }],
          };
        }
        let result;
        try {
          result = await apiRequest("POST", "/api/memories", {
            content: params.content,
            category: params.category || "facts",
            tags: params.tags || "",
            expanded_keywords: params.expanded_keywords || "",
            importance: params.importance ?? 0.5,
          });
        } catch (err) {
          if (err?.status === 422) {
            // Surface the server's guidance as a readable tool result, not a crash.
            return {
              content: [{ type: "text", text: `Memory rejected by server: ${err.message}` }],
            };
          }
          throw err;
        }
        return {
          content: [
            {
              type: "text",
              text: `Stored memory #${result.id} in category '${result.category}' with importance ${Number(result.importance).toFixed(1)}`,
            },
          ],
        };
      },
    });

    // memory_recall
    api.registerTool({
      name: "memory_recall",
      label: "Memory Recall",
      description:
        "Retrieve relevant memories based on context. Uses full-text search to find stored memories.",
      parameters: {
        type: "object",
        properties: {
          context: {
            type: "string",
            description: "The context or topic to recall memories about",
          },
          expanded_query: {
            type: "string",
            description:
              "REQUIRED. Space-separated semantically related search terms (MINIMUM 5 words).",
          },
          category: {
            type: "string",
            enum: ["facts", "preferences", "projects", "people", "decisions"],
            description: "Optional: filter results to a specific category",
          },
          sort_by: {
            type: "string",
            enum: ["importance", "relevance"],
            description: "Sort order (default: relevance)",
          },
          limit: { type: "integer", description: "Max results" },
        },
        required: ["context", "expanded_query"],
      },
      async execute(_toolCallId, params) {
        // Default flipped from "importance" per the ADR-0005 amendment.
        const sortBy = params.sort_by || "relevance";
        const result = await apiRequest("POST", "/api/memories/recall", {
          context: params.context,
          expanded_query: params.expanded_query || "",
          category: params.category || null,
          sort_by: sortBy,
          limit: params.limit || 10,
        });
        const rows = result.memories || [];
        if (!rows.length) {
          const filterDesc = params.category ? ` in category '${params.category}'` : "";
          return {
            content: [
              { type: "text", text: `No memories found matching: ${params.context}${filterDesc}` },
            ],
          };
        }
        const sortDesc = sortBy === "relevance" ? "by relevance" : "by importance";
        const filterDesc = params.category ? ` in '${params.category}'` : "";
        const lines = rows.map(
          (r) =>
            `#${r.id} [${r.category}] (importance: ${Number(r.importance).toFixed(1)}) ${r.content}\n  Tags: ${r.tags || "none"} | Stored: ${r.created_at}`,
        );
        return {
          content: [
            {
              type: "text",
              text: `Found ${rows.length} memories${filterDesc} (${sortDesc}):\n\n${lines.join("\n\n")}`,
            },
          ],
        };
      },
    });

    // memory_list
    api.registerTool({
      name: "memory_list",
      label: "Memory List",
      description: "List recent memories, optionally filtered by category.",
      parameters: {
        type: "object",
        properties: {
          category: {
            type: "string",
            enum: ["facts", "preferences", "projects", "people", "decisions"],
          },
          limit: { type: "integer" },
        },
      },
      async execute(_toolCallId, params) {
        const limit = params.limit || 20;
        let path = `/api/memories?limit=${limit}`;
        if (params.category) {
          path += `&category=${params.category}`;
        }
        const result = await apiRequest("GET", path);
        const rows = result.memories || [];
        if (!rows.length) {
          return {
            content: [
              {
                type: "text",
                text: params.category
                  ? `No memories in category '${params.category}'`
                  : "No memories stored yet",
              },
            ],
          };
        }
        const lines = rows.map(
          (r) =>
            `#${r.id} [${r.category}] ${r.content}\n  Importance: ${Number(r.importance).toFixed(1)} | Tags: ${r.tags || "none"} | Stored: ${r.created_at}`,
        );
        const header =
          "Recent memories" + (params.category ? ` in '${params.category}'` : "");
        return {
          content: [
            {
              type: "text",
              text: `${header} (${rows.length} shown):\n\n${lines.join("\n\n")}`,
            },
          ],
        };
      },
    });

    // memory_delete
    api.registerTool({
      name: "memory_delete",
      label: "Memory Delete",
      description: "Delete a memory by ID.",
      parameters: {
        type: "object",
        properties: {
          id: { type: "integer", description: "The ID of the memory to delete" },
        },
        required: ["id"],
      },
      async execute(_toolCallId, params) {
        const result = await apiRequest("DELETE", `/api/memories/${params.id}`);
        return {
          content: [
            {
              type: "text",
              text: `Deleted memory #${result.deleted}: ${result.preview}...`,
            },
          ],
        };
      },
    });
  },
};

export default memoryApiPlugin;
