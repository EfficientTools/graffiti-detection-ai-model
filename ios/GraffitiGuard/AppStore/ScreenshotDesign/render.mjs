#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const designDir = dirname(fileURLToPath(import.meta.url));
const appStoreDir = resolve(designDir, "..");
const outputDir = join(appStoreDir, "Screenshots", "en-US");
const sourceDir = join(designDir, "source");
const iconPath = resolve(
  appStoreDir,
  "..",
  "GraffitiGuard",
  "Resources",
  "Assets.xcassets",
  "AppIcon.appiconset",
  "AppIcon.png",
);

const slides = [
  {
    slug: "spot-fast",
    title: ["Spot graffiti.", "Fast."],
    body: "On-device AI highlights likely graffiti in a single tap.",
    badge: "SMART DETECTION",
    source: "detection",
    accent: "#B8FF46",
    background: ["#091713", "#183C2E"],
    tilt: -1.4,
  },
  {
    slug: "stay-private",
    title: ["Your photos", "stay private."],
    body: "Images are processed on your device and never uploaded.",
    badge: "PRIVACY FIRST",
    source: "privacy",
    accent: "#36E6C2",
    background: ["#07171A", "#124248"],
    tilt: 1.2,
  },
  {
    slug: "no-server",
    title: ["Works without", "a server."],
    body: "No account. No cloud. No waiting.",
    badge: "OFFLINE CORE ML",
    source: "ready",
    accent: "#FFB449",
    background: ["#1A120A", "#4A3116"],
    tilt: -0.8,
  },
  {
    slug: "control-confidence",
    title: ["Confidence", "you control."],
    body: "Tune the detection threshold for every inspection.",
    badge: "ADJUSTABLE",
    source: "privacy",
    accent: "#FF765F",
    background: ["#1C0E10", "#4A1D21"],
    tilt: 1.1,
  },
  {
    slug: "photo-to-answer",
    title: ["From photo", "to answer."],
    body: "Choose an image, detect, and review the highlighted area.",
    badge: "THREE SIMPLE STEPS",
    source: "ready",
    accent: "#84A9FF",
    background: ["#0B1122", "#213B6B"],
    tilt: -1.2,
  },
  {
    slug: "real-streets",
    title: ["Built for", "real streets."],
    body: "Clear results support faster maintenance decisions.",
    badge: "FIELD READY",
    source: "detection",
    accent: "#B8FF46",
    background: ["#0B1411", "#294536"],
    tilt: 0.7,
  },
];

const tempDir = mkdtempSync(join(tmpdir(), "graffiti-store-shots-"));

function dataUrl(path) {
  return `data:image/png;base64,${readFileSync(path).toString("base64")}`;
}

function escapeXml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function sharedDefinitions(slide, iconData) {
  return `
    <defs>
      <linearGradient id="background" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="${slide.background[0]}" />
        <stop offset="1" stop-color="${slide.background[1]}" />
      </linearGradient>
      <radialGradient id="glow" cx="50%" cy="50%" r="50%">
        <stop offset="0" stop-color="${slide.accent}" stop-opacity="0.32" />
        <stop offset="1" stop-color="${slide.accent}" stop-opacity="0" />
      </radialGradient>
      <pattern id="grid" width="92" height="92" patternUnits="userSpaceOnUse" patternTransform="rotate(18)">
        <path d="M 92 0 L 0 0 0 92" fill="none" stroke="#FFFFFF" stroke-opacity="0.045" stroke-width="2" />
      </pattern>
      <filter id="shadow" x="-30%" y="-30%" width="160%" height="180%">
        <feDropShadow dx="0" dy="34" stdDeviation="35" flood-color="#000000" flood-opacity="0.48" />
      </filter>
      <filter id="softShadow" x="-30%" y="-30%" width="160%" height="180%">
        <feDropShadow dx="0" dy="16" stdDeviation="18" flood-color="#000000" flood-opacity="0.32" />
      </filter>
    </defs>
    <rect width="100%" height="100%" fill="url(#background)" />
    <rect width="100%" height="100%" fill="url(#grid)" />
    <circle cx="94%" cy="17%" r="420" fill="url(#glow)" />
    <circle cx="7%" cy="87%" r="520" fill="url(#glow)" opacity="0.62" />
    <g opacity="0.52">
      <circle cx="87%" cy="21%" r="13" fill="${slide.accent}" />
      <circle cx="91%" cy="25%" r="7" fill="${slide.accent}" />
      <circle cx="84%" cy="26%" r="5" fill="${slide.accent}" />
    </g>
    <g transform="translate(82 76)">
      <rect width="72" height="72" rx="18" fill="#F4F8F1" />
      <image href="${iconData}" x="4" y="4" width="64" height="64" />
      <text x="92" y="47" fill="#FFFFFF" font-family="Avenir Next, Avenir, sans-serif" font-size="30" font-weight="700" letter-spacing="2">GRAFFITI GUARD</text>
    </g>
  `;
}

function headline(slide, width, titleSize, bodySize, titleY, lineGap) {
  const right = width - 82;
  const badgeWidth = Math.max(260, slide.badge.length * 20 + 66);
  return `
    <g font-family="Avenir Next, Avenir, sans-serif">
      <rect x="${right - badgeWidth}" y="82" width="${badgeWidth}" height="58" rx="29" fill="${slide.accent}" fill-opacity="0.16" stroke="${slide.accent}" stroke-opacity="0.75" stroke-width="2" />
      <text x="${right - badgeWidth / 2}" y="120" text-anchor="middle" fill="${slide.accent}" font-size="22" font-weight="700" letter-spacing="2.6">${escapeXml(slide.badge)}</text>
      <text x="82" y="${titleY}" fill="#FFFFFF" font-size="${titleSize}" font-weight="700" letter-spacing="-3.5">${escapeXml(slide.title[0])}</text>
      <text x="82" y="${titleY + lineGap}" fill="${slide.accent}" font-size="${titleSize}" font-weight="700" letter-spacing="-3.5">${escapeXml(slide.title[1])}</text>
      <text x="86" y="${titleY + lineGap + 100}" fill="#E7EEE9" fill-opacity="0.9" font-size="${bodySize}" font-weight="500">${escapeXml(slide.body)}</text>
    </g>
  `;
}

function iphoneSvg(slide, index, iconData, screenshotData) {
  const width = 1320;
  const height = 2868;
  const screenX = 190;
  const screenY = 780;
  const screenWidth = 940;
  const screenHeight = 2042;

  return `<?xml version="1.0" encoding="UTF-8"?>
  <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    ${sharedDefinitions(slide, iconData)}
    ${headline(slide, width, 126, 38, 304, 132)}
    <g transform="rotate(${slide.tilt} 660 1800)" filter="url(#shadow)">
      <rect x="153" y="743" width="1014" height="2116" rx="134" fill="#050908" stroke="#FFFFFF" stroke-opacity="0.2" stroke-width="4" />
      <rect x="174" y="764" width="972" height="2078" rx="114" fill="#E9F0EA" />
      <clipPath id="iphone-screen-${index}">
        <rect x="${screenX}" y="${screenY}" width="${screenWidth}" height="${screenHeight}" rx="98" />
      </clipPath>
      <image href="${screenshotData}" x="${screenX}" y="${screenY}" width="${screenWidth}" height="${screenHeight}" preserveAspectRatio="none" clip-path="url(#iphone-screen-${index})" />
    </g>
    <g transform="translate(76 689)" filter="url(#softShadow)">
      <rect width="194" height="58" rx="29" fill="#F5F8F4" />
      <circle cx="31" cy="29" r="9" fill="${slide.accent}" />
      <text x="52" y="38" fill="#111B18" font-family="Avenir Next, Avenir, sans-serif" font-size="22" font-weight="700">0${index + 1} / 06</text>
    </g>
  </svg>`;
}

function ipadSvg(slide, index, iconData, screenshotData) {
  const width = 2064;
  const height = 2752;
  const screenX = 130;
  const screenY = 716;
  const screenWidth = 1804;
  const visibleHeight = 1902;
  const scaledHeight = Math.round((screenWidth * 2752) / 2064);

  return `<?xml version="1.0" encoding="UTF-8"?>
  <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    ${sharedDefinitions(slide, iconData)}
    ${headline(slide, width, 118, 38, 294, 122)}
    <g transform="rotate(${slide.tilt * 0.35} 1032 1680)" filter="url(#shadow)">
      <rect x="82" y="668" width="1900" height="2010" rx="74" fill="#050908" stroke="#FFFFFF" stroke-opacity="0.2" stroke-width="4" />
      <rect x="108" y="694" width="1848" height="1958" rx="52" fill="#E9F0EA" />
      <clipPath id="ipad-screen-${index}">
        <rect x="${screenX}" y="${screenY}" width="${screenWidth}" height="${visibleHeight}" rx="38" />
      </clipPath>
      <image href="${screenshotData}" x="${screenX}" y="${screenY}" width="${screenWidth}" height="${scaledHeight}" preserveAspectRatio="none" clip-path="url(#ipad-screen-${index})" />
    </g>
    <g transform="translate(110 612)" filter="url(#softShadow)">
      <rect width="194" height="58" rx="29" fill="#F5F8F4" />
      <circle cx="31" cy="29" r="9" fill="${slide.accent}" />
      <text x="52" y="38" fill="#111B18" font-family="Avenir Next, Avenir, sans-serif" font-size="22" font-weight="700">0${index + 1} / 06</text>
    </g>
  </svg>`;
}

function render(svg, outputPath, width, height) {
  const svgPath = join(tempDir, `${outputPath.split("/").at(-1)}.svg`);
  const rgbaPath = join(tempDir, `${outputPath.split("/").at(-1)}.rgba.png`);
  writeFileSync(svgPath, svg);
  execFileSync("rsvg-convert", ["--width", String(width), "--height", String(height), "--output", rgbaPath, svgPath]);
  execFileSync("magick", [rgbaPath, "-strip", "-colorspace", "sRGB", "-background", "#0B1512", "-alpha", "remove", "-alpha", "off", `PNG24:${outputPath}`]);
}

mkdirSync(outputDir, { recursive: true });
const iconData = dataUrl(iconPath);

try {
  for (const [index, slide] of slides.entries()) {
    for (const platform of ["iphone", "ipad"]) {
      const screenshotData = dataUrl(join(sourceDir, `${platform}-${slide.source}.png`));
      const isPhone = platform === "iphone";
      const outputName = isPhone
        ? `iphone-6.9-0${index + 1}-${slide.slug}.png`
        : `ipad-13-0${index + 1}-${slide.slug}.png`;
      const outputPath = join(outputDir, outputName);
      const svg = isPhone
        ? iphoneSvg(slide, index, iconData, screenshotData)
        : ipadSvg(slide, index, iconData, screenshotData);
      render(svg, outputPath, isPhone ? 1320 : 2064, isPhone ? 2868 : 2752);
      console.log(outputPath);
    }
  }
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}
