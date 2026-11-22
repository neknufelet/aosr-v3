"use strict";

function catalogKind(entry) {
  return entry.kind === "fixed" || (entry.kind === "construction" && entry.product) ? "fixed" : "optimizable";
}
