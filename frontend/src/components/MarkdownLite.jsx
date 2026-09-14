import React from "react";

/** Minimal renderer for the short Markdown the agent produces: paragraphs, bullet lists,
 * fenced code blocks, and **bold** spans. Not a general Markdown parser. */
export default function MarkdownLite({ text }) {
  const blocks = splitBlocks(text || "");
  return (
    <div className="markdown-lite">
      {blocks.map((block, index) => {
        if (block.type === "code") {
          return (
            <pre key={index}>
              <code>{block.content}</code>
            </pre>
          );
        }
        if (block.type === "list") {
          return (
            <ul key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>{renderInline(item)}</li>
              ))}
            </ul>
          );
        }
        return <p key={index}>{renderInline(block.content)}</p>;
      })}
    </div>
  );
}

function splitBlocks(text) {
  const lines = text.split("\n");
  const blocks = [];
  let paragraph = [];
  let listItems = [];
  let inCode = false;
  let codeLines = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: "text", content: paragraph.join(" ") });
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listItems.length) {
      blocks.push({ type: "list", items: listItems });
      listItems = [];
    }
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCode) {
        blocks.push({ type: "code", content: codeLines.join("\n") });
        codeLines = [];
      } else {
        flushParagraph();
        flushList();
      }
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)/);
    if (bullet) {
      flushParagraph();
      listItems.push(bullet[1]);
      continue;
    }
    flushList();
    if (line.trim() === "") {
      flushParagraph();
    } else {
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  flushList();
  if (inCode && codeLines.length) blocks.push({ type: "code", content: codeLines.join("\n") });
  return blocks;
}

function renderInline(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={index}>{part.slice(2, -2)}</strong> : part
  );
}
