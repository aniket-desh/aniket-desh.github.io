#!/usr/bin/env node
/*
 * Typeset the math in built pages at deploy time.
 *
 *   node scripts/prerender_math.js <built-site-dir>
 *
 * Pages are written with TeX between \( \) and \[ \] and load MathJax from a
 * CDN, so they preview locally with no build step. On a slow phone that means
 * seconds of raw TeX while the 205 KB script downloads and runs. This script
 * runs the same MathJax version over the built copy of the site instead: each
 * expression is replaced by MathJax's CommonHTML output, the stylesheet it
 * needs is inlined in <head>, and the MathJax <script> is removed. The source
 * files in the repository are never touched.
 *
 * Everything outside the math is left byte-for-byte as it was. The page is not
 * re-serialized through a DOM, so inline scripts, SVG and entities are safe.
 *
 * A page is left alone (and keeps client-side MathJax) when it
 *   - does not load MathJax,
 *   - calls MathJax.typeset itself, because it adds math after load, or
 *   - sets a MathJax config with macros, packages or a loader, which this
 *     script does not read.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {CHTML} = require('mathjax-full/js/output/chtml.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AssistiveMmlHandler} = require('mathjax-full/js/a11y/assistive-mml.js');
require('mathjax-full/js/input/tex/AllPackages.js'); // registers every TeX package

const MATHJAX_VERSION = require('mathjax-full/package.json').version;
const FONT_URL = `https://cdn.jsdelivr.net/npm/mathjax@${MATHJAX_VERSION}/es5/output/chtml/fonts/woff-v2`;

// The packages the tex-mml-chtml.js bundle loads up front, plus the ones it
// autoloads on first use, so a page typesets the same here as in the browser.
const PACKAGES = [
  'base', 'ams', 'newcommand', 'noundefined', 'configmacros',
  'action', 'amscd', 'bbox', 'boldsymbol', 'braket', 'bussproofs', 'cancel',
  'color', 'enclose', 'extpfeil', 'html', 'mhchem', 'unicode', 'verb',
];

// MathJax scales math so its x-height matches the surrounding text. In the
// browser it measures the text font when it happens to run: 103.3% once
// Alegreya has loaded (x-height 0.4566 em, against 0.442 em for the MathJax TeX
// font), but 108.8% if it runs first and measures the Georgia fallback. Here
// the scale is fixed at the Alegreya value, so math is the same size on every
// load.
const EM = 16;
const EX = 0.4566 * EM;
const SCALE = Math.round((EX / EM / 0.442) * 1000) / 1000; // 1.033

const adaptor = liteAdaptor();
AssistiveMmlHandler(RegisterHTMLHandler(adaptor));

const LOADER_TAG = /[ \t]*<script\b[^>]*\bsrc="[^"]*\bmathjax@[^"]*"[^>]*>\s*<\/script>[ \t]*\n?/i;
const CONFIG_BLOCK = /[ \t]*<script>\s*window\.MathJax\s*=\s*\{[\s\S]*?\};?\s*<\/script>[ \t]*\n?/;

// Regions MathJax never looks inside: comments, the elements on its skip list,
// and the tags themselves (an attribute value may contain a delimiter).
const PROTECTED = new RegExp(
  '<!--[\\s\\S]*?-->' +
  '|<(script|style|pre|code|textarea|noscript|annotation|annotation-xml)\\b[\\s\\S]*?<\\/\\1\\s*>' +
  '|<\\/?[a-zA-Z](?:"[^"]*"|\'[^\']*\'|[^>"\'])*>',
  'gi'
);

// \[ ... \] and $$ ... $$ are display math, \( ... \) is inline: the MathJax defaults.
const MATH = /\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g;

const ENTITIES = {amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' '};

function decodeEntities(text) {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, name) => {
    if (name[0] !== '#') return Object.hasOwn(ENTITIES, name) ? ENTITIES[name] : whole;
    const code = name[1].toLowerCase() === 'x' ? parseInt(name.slice(2), 16) : parseInt(name.slice(1), 10);
    return String.fromCodePoint(code);
  });
}

function skipReason(html) {
  if (!LOADER_TAG.test(html)) return 'no MathJax';
  if (/MathJax\.typeset/.test(html)) return 'typesets math itself';
  const config = html.match(CONFIG_BLOCK);
  if (config && /\b(macros|packages|loader)\b/.test(config[0])) return 'custom MathJax config';
  return null;
}

function prerender(html) {
  const tex = new TeX({packages: PACKAGES});
  const chtml = new CHTML({fontURL: FONT_URL});
  const doc = mathjax.document('', {InputJax: tex, OutputJax: chtml});

  let count = 0;
  const errors = [];
  const typeset = (source, display) => {
    const node = doc.convert(decodeEntities(source), {display, em: EM, ex: EX, scale: SCALE, containerWidth: 80 * EM});
    const out = adaptor.outerHTML(node);
    if (out.includes('data-mjx-error')) errors.push(source.trim().replace(/\s+/g, ' ').slice(0, 80));
    count += 1;
    return out;
  };
  const typesetText = (text) =>
    text.replace(MATH, (whole, bracket, dollars, paren) => {
      const display = paren === undefined;
      return typeset(display ? (bracket !== undefined ? bracket : dollars) : paren, display);
    });

  const bodyStart = html.search(/<body\b/i);
  if (bodyStart < 0) return null;
  const body = html.slice(bodyStart);

  let out = '';
  let last = 0;
  for (const match of body.matchAll(PROTECTED)) {
    out += typesetText(body.slice(last, match.index)) + match[0];
    last = match.index + match[0].length;
  }
  out += typesetText(body.slice(last));

  let head = html.slice(0, bodyStart).replace(CONFIG_BLOCK, '');
  if (count === 0) {
    // The page loads MathJax but has nothing to typeset.
    return {html: head.replace(LOADER_TAG, '') + out, count, errors};
  }
  const indent = (head.match(/^([ \t]*)<script\b[^>]*mathjax@/im) || ['', '    '])[1];
  const css = adaptor.textContent(chtml.styleSheet(doc));
  head = head.replace(
    LOADER_TAG,
    `${indent}<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>\n` +
    `${indent}<style id="MJX-CHTML-styles">${css}</style>\n`
  );
  return {html: head + out, count, errors};
}

function* htmlFiles(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'node_modules' && entry.name !== '.git') yield* htmlFiles(full);
    } else if (entry.name.endsWith('.html')) {
      yield full;
    }
  }
}

function main() {
  const root = process.argv[2];
  if (!root || !fs.statSync(root, {throwIfNoEntry: false})?.isDirectory()) {
    console.error('usage: node scripts/prerender_math.js <built-site-dir>');
    process.exit(2);
  }

  let pages = 0;
  let expressions = 0;
  let failed = 0;
  for (const file of htmlFiles(root)) {
    const name = path.relative(root, file);
    const html = fs.readFileSync(file, 'utf8');
    const reason = skipReason(html);
    if (reason) {
      if (reason !== 'no MathJax') console.log(`skip  ${name} (${reason})`);
      continue;
    }
    try {
      const result = prerender(html);
      if (!result) continue;
      fs.writeFileSync(file, result.html);
      pages += 1;
      expressions += result.count;
      console.log(`${String(result.count).padStart(5)} ${name}`);
      for (const source of result.errors) console.log(`      TeX error in: ${source}`);
    } catch (error) {
      // Leave the page as it was: it still loads MathJax in the browser.
      failed += 1;
      console.log(`FAIL  ${name}: ${error.message}`);
    }
  }
  console.log(`pre-rendered ${expressions} expressions on ${pages} pages` + (failed ? `, ${failed} pages left on client-side MathJax` : ''));
}

main();
