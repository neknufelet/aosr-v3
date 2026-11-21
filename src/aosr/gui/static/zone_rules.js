/* 「其他牆面」那條規則怎麼從各區推（#1176 第 C 支）。設計檔每一區只存一個做法（fixed／keep_original／search），沒有
   「跟著規則」這一格；畫面上那條「其他牆面」是每次從各區現算的，改規則時再展開寫回每一區。這一份只有純函式，不碰頁面。

   算進來的區：前、後、左、右四面牆的區，扣掉門、窗（呼叫的人先濾好，照「前牆、後牆、左牆、右牆」、每面照號碼排好再交進來）。
   每一區一個「做法鍵」（zoneMethodKey）：
   - 由程式最佳化：程式可以選的那幾種當集合比（順序、重複不算）；
   - 不貼，保留原樣：一個鍵；
   - 指定貼一種：這一區現在是材料庫哪一筆；認不到哪一筆就拿貼上去的那幾層比——指定貼不同的材料算不同的做法；
   - 指定貼、但材料就是原牆原樣（設計檔沒列到的區照規矩當指定貼，可是上面什麼都沒貼）：跟「不貼，保留原樣」同一個鍵
     （老闆看不出差別，畫面上也叫不貼）。只是推法與顯示這樣併，設計檔裡那一區的做法不會因此被改寫。
   規則＝出現最多次的做法鍵；平手取交進來的順序裡最先出現的那一個。另外指定＝做法鍵跟規則不同的區，同一個鍵一組，
   組照最先出現的順序排。

   清單卡（zoneSearchList）：由程式最佳化的每一區各存一份清單，清單一樣的區一組、一組一張卡（groups）。排第一的那一組
   （common）：規則本身是由程式最佳化就是規則那一組，不然是最多區用的那一組（平手取最先出現的）；其餘照最先出現的順序。
   清單跟第一組不同的區另外數出來（different）。
   每一區自己的可調範圍（#1199，設計檔的 ranges）只拆清單卡、不進做法鍵：清單一樣、範圍不同的區分成兩張卡（範圍跟「這一區最深」
   一樣不算另一種做法，「其他牆面」怎麼推、哪幾區算另外指定都照舊）。規則那一組被範圍拆開時，排第一的是其中區數最多的那一張。 */

// 物件的鍵排過序再寫成字串（值是 null 的鍵當沒有）：兩份材料內容一樣，字串就一樣。
function zoneCanonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(zoneCanonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const keys = Object.keys(value).filter(key => value[key] !== null && value[key] !== undefined).sort();
    return `{${keys.map(key => `${JSON.stringify(key)}:${zoneCanonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
// 指定貼的是哪一種：材料庫認得就是那一筆的代號（libraryId）；認不到，貼上去的比那幾層、沒貼東西的比整份材料。
function zoneFixedIdentity(material, libraryId) {
  if (libraryId) return `library:${libraryId}`;
  return material.kind === "mounted_on_wall" ? `layers:${zoneCanonicalJson(material.layers)}` : `bare:${zoneCanonicalJson(material)}`;
}
// 一區的做法鍵。zone：{mode, allowed（由程式最佳化的清單；讀不出是哪幾種時 null）, fixed（zoneFixedIdentity 的結果）}。
function zoneMethodKey(zone) {
  if (zone.mode === "search") return `search:${zone.allowed ? [...new Set(zone.allowed)].sort().join(",") : "?"}`;
  if (zone.construction?.state === "unknown") return `unknown:${zone.construction.id}`;
  if (zone.mode === "keep_original" || zone.fixed.startsWith("bare:")) return "keep_original";
  return `fixed:${zone.fixed}`;
}
// 做法鍵是哪一種做法（畫面上叫什麼、畫什麼顏色用它，不直接用設計檔那一格）。
function zoneMethodMode(key) { return key === "keep_original" ? "keep_original" : key.split(":")[0]; }
// 照做法鍵分組，組照最先出現的順序排：[{key, mode, zones}]。
function zoneMethodGroups(zones) {
  const groups = new Map();
  for (const zone of zones) {
    const key = zoneMethodKey(zone);
    if (!groups.has(key)) groups.set(key, {key, mode: zoneMethodMode(key), zones: []});
    groups.get(key).zones.push(zone);
  }
  return [...groups.values()];
}
// 規則與另外指定：{rule: 那一組或 null（一區都沒有）, special: [其他組]}。平手時先出現的那一組贏（只有「更多」才換）。
// keptKey：這次打開之後已經定下來的規則（做法鍵）；還有區照著它，規則就是它，不管哪一組區數比較多——
// 使用者一區一區改到別組比較多時，「其他牆面」不會自己換。沒有區照著它了（或沒給）才照區數推。
function inferOtherWalls(zones, keptKey = null) {
  const groups = zoneMethodGroups(zones);
  let rule = groups.find(group => group.key === keptKey) ?? null;
  if (!rule) for (const group of groups) if (!rule || group.zones.length > rule.zones.length) rule = group;
  return {rule, special: groups.filter(group => group !== rule)};
}
// 清單卡：{zones: 全部由程式最佳化的區, common: 排第一的那一組或 null, different: 清單跟它不同的區, groups: 每一組（common 在最前）}。
// ruleKey：規則的做法鍵；規則是由程式最佳化、而且真的有區用那一份時，列的就是它。
// 一區自己的可調範圍寫成一個鍵（#1199）：沒寫是空字串；材料代號與欄位排過序，數字一樣字串就一樣。
function zoneRangesKey(zone) { return zone.ranges ? zoneCanonicalJson(zone.ranges) : ""; }
function zoneSearchList(zones, ruleKey) {
  const search = zones.filter(zone => zone.mode === "search");
  // 清單一樣的那一組再照各區自己的範圍拆開（照最先出現的順序；沒寫範圍的區照舊一組）。
  const groups = zoneMethodGroups(search).flatMap(group => {
    const parts = new Map();
    for (const zone of group.zones) parts.set(zoneRangesKey(zone), [...(parts.get(zoneRangesKey(zone)) ?? []), zone]);
    return [...parts.values()].map(members => ({...group, zones: members}));
  });
  // 區數最多的那一張（平手取最先出現的，同 inferOtherWalls）。
  const biggest = list => list.reduce((best, group) => !best || group.zones.length > best.zones.length ? group : best, null);
  const common = biggest(groups.filter(group => group.key === ruleKey)) ?? biggest(groups);
  return {zones: search, common, different: common ? search.filter(zone => !common.zones.includes(zone)) : [],
          groups: common ? [common, ...groups.filter(group => group !== common)] : []};
}
