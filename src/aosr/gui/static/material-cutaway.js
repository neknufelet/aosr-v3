"use strict";

// 房間側在左，牆側在右。以毫米排寬度，薄板與布至少留一條看得到的邊。
function materialSvg(tag, attributes) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  return node;
}

function materialTexture(kind, materialClass) {
  if (kind === "porous") return materialClass === "foam" ? "foam" : "wool";
  return {micro_perforated: "mpp", perforated: "perf", slotted: "slot"}[kind] || kind;
}

function materialCutaway(layers, backing, description) {
  const svg = materialSvg("svg", {class: "cutaway", role: "img", "aria-label": "剖面：" + description});
  const title = materialSvg("title", {});
  title.textContent = description;
  svg.append(title);
  const slabs = layers.map(layer => ({...layer, width: Math.max(3, (4 + 3 * Math.pow(layer.mm, .8)) * .33)}));
  if (backing === "rigid") slabs.push({texture: "wall", width: 8});
  if (!slabs.length) {
    svg.setAttribute("viewBox", "0 0 96 68");
    svg.setAttribute("width", 96);
    svg.append(materialSvg("path", {d: "M5 8 L40 8 L52 15 V42 H17 L5 35 Z M5 8 L17 15 H52 M17 15 V42",
      fill: "#f6f8f9", stroke: "#839399", "stroke-dasharray": "3 2"}));
    const unknown = materialSvg("text", {x: 31, y: 35, fill: "#53636b", "font-size": 17});
    unknown.textContent = "?";
    svg.append(unknown);
    const label = materialSvg("text", {x: 4, y: 62, fill: "#53636b", "font-size": 13});
    label.textContent = "無分層資料";
    svg.append(label);
    return svg;
  }
  const bottom = 50 + slabs.length * 6;
  let right = 30;
  for (const slab of slabs) {
    slab.x = right;
    right += slab.width;
  }
  svg.setAttribute("viewBox", `0 0 ${right + 4} ${bottom + 3}`);
  svg.setAttribute("width", right + 4);
  for (let i = slabs.length - 1; i >= 0; i--) {
    const {x, width, texture, adjustable} = slabs[i];
    const top = bottom - 34 - i * 6;
    const group = materialSvg("g", {stroke: adjustable ? "#c05a11" : "#52646d",
      "stroke-width": adjustable ? 1.8 : .9, "stroke-linejoin": "round",
      fill: texture === "air" ? "white" : `url(#mat-${texture})`});
    group.append(materialSvg("polygon", {points: `${x},${bottom} ${x},${top} ${x-26},${top-13} ${x-26},${bottom-13}`}),
      materialSvg("polygon", {points: `${x},${top} ${x+width},${top} ${x+width-26},${top-13} ${x-26},${top-13}`}),
      materialSvg("rect", {x, y: top, width, height: bottom-top}));
    if (texture === "air") group.setAttribute("fill-opacity", ".7");
    svg.append(group);
  }
  return svg;
}

// 共用剖面使用的材質花紋，整頁只需一份。
document.body.insertAdjacentHTML("afterbegin", `<svg aria-hidden="true" style="position:absolute;width:0;height:0;border:0"><defs>
<pattern id="mat-wool" width="16" height="8" patternUnits="userSpaceOnUse"><rect width="16" height="8" fill="#f4e7b4"/><path d="M0 4 q4 -5 8 0 t8 0" fill="none" stroke="#a8862a" stroke-width="1.1"/></pattern>
<pattern id="mat-foam" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#e6ecef"/><circle cx="2" cy="2" r="1.1" fill="#748a93"/><circle cx="6" cy="6" r="1.1" fill="#748a93"/></pattern>
<pattern id="mat-board" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#cdbb9c"/></pattern>
<pattern id="mat-steel" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#87949b"/></pattern>
<pattern id="mat-fabric" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" fill="#dfe7e1"/><path d="M0 0 L5 5 M5 0 L0 5" stroke="#8fa397" stroke-width=".6"/></pattern>
<pattern id="mat-mpp" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" fill="#cdbb9c"/><circle cx="2.5" cy="2.5" r=".75" fill="#3a2f20"/></pattern>
<pattern id="mat-perf" width="13" height="13" patternUnits="userSpaceOnUse"><rect width="13" height="13" fill="#cdbb9c"/><circle cx="6.5" cy="6.5" r="2.7" fill="#3a2f20"/></pattern>
<pattern id="mat-slot" width="14" height="46" patternUnits="userSpaceOnUse"><rect width="14" height="46" fill="#cdbb9c"/><rect x="5.5" y="4" width="3" height="38" rx="1.5" fill="#3a2f20"/></pattern>
<pattern id="mat-wall" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#c9cfd3"/><path d="M-2 2 L2 -2 M0 8 L8 0 M6 10 L10 6" stroke="#6f7c83" stroke-width="1.2"/></pattern>
</defs></svg>`);
