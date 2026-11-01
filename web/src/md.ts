// Tiny safe markdown subset -> HTML: #/##/### headings, paragraphs, "- " / "* " bullet lists, **bold**, `code`, [text](http(s) url).
// Everything is HTML-escaped first, so the output can be assigned to innerHTML.
export const escapeHtml = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

function inline(raw: string): string {
  let s = escapeHtml(raw);
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)"']+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s;
}

export function renderMarkdown(md: string): string {
  const out: string[] = [];
  let para: string[] = [], list: string[] = [];
  const flushP = () => { if (para.length) out.push(`<p>${inline(para.join(' '))}</p>`); para = []; };
  const flushL = () => { if (list.length) out.push(`<ul>${list.map(i => `<li>${inline(i)}</li>`).join('')}</ul>`); list = []; };
  for (const line of md.replace(/\r/g, '').split('\n')) {
    const h = /^(#{1,3})\s+(.*)$/.exec(line), li = /^\s*[-*]\s+(.*)$/.exec(line);
    if (h) { flushP(); flushL(); const n = h[1].length + 2; out.push(`<h${n}>${inline(h[2])}</h${n}>`); }
    else if (li) { flushP(); list.push(li[1]); }
    else if (!line.trim()) { flushP(); flushL(); }
    else if (list.length && /^\s+/.test(line)) list[list.length - 1] += ' ' + line.trim(); // wrapped bullet
    else { flushL(); para.push(line.trim()); }
  }
  flushP(); flushL();
  return out.join('');
}
