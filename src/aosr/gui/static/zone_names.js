/* 區名與號碼（#1176 第 B 支，老闆 11-15「自動 auto3 看不出是牆上哪一塊」）：一面牆上每一區一個號碼①②③…和一句位置名
   （「左聲道喇叭後面那塊」「靠天花那條」），立面圖與做法表共用這一份（立面圖那邊把畫面座標交進來，這裡只算字）。
   伺服器的訊息與搜尋頁還在用區代號（z1、auto3），所以表上名字底下另外寫小字代號。

   座標一律是畫面上看到的方向（站在房間裡看那一面；鏡像的牆已經翻好）：a 從左緣往右、b 從下緣往上，公尺。
   號碼：先畫的框、再自動補的格子，各自照畫面由上到下、同一排由左到右（上緣高的先，上緣一樣高的左緣靠左的先）。
   名字規則有先後（前面的成立就用它）：
   ⓪ 材料是門、窗、地毯：「門」「窗」「地毯」（材料庫分組 door_window 且顯示名含門／窗；分組 floor 且代號含 carpet）；
   ① 蓋住喇叭的投影：前牆「左聲道喇叭後面那塊」、地板「…底下那塊」、天花「…上面那塊」——只用在這三面
      （兩側牆、後牆也投影得到，但那裡不是喇叭「後面」；兩支喇叭在側牆上投影在同一點，叫哪一支都不對）；
   ② 蓋住主位的投影：前牆「主位正前方」、後牆「主位正後方」、地板「主位底下」、天花「主位上面」；
   ③ 整面一區：「整面」；
   ④ 長條（橫跨整面寬、直跨整面高，或長邊／短邊夠長的條）：長邊貼著哪一邊就「靠＜那一邊隔壁那一面＞那條」
      （隔壁那一面的名字就是立面圖四邊印的牆名：前牆的左緣是左牆、後牆是鏡像所以左緣是右牆、地板天花的四邊是四面牆）；
      不貼邊的區（長條或不是）兩側夾著兩塊有名字的（⓪～②）就「A 和 B 中間那條／那塊」（兩支喇叭就寫「兩支喇叭中間那條」）；
      橫跨或直跨整面、不貼邊也沒夾著的寫「中間那條」；
   ⑤ 其他照中心落在三等分：牆「左上」「正中間」「右下」…；地板天花「前左」「後右」…（前＝前牆那一側，左＝左牆那一側；
      牆名定不出前後左右時照畫面方向）。
   長邊／短邊夠長叫「條」，否則「塊」。同名的兩區名字照留，靠號碼分。 */

// 長邊／短邊到這個倍數算「條」（也是④不必橫跨整面就算長條的門檻）。命名用的字形門檻，不是物理或評分的數字。
const ZONE_STRIP_RATIO = 3;
const ZONE_GRID_WALL = [["左上", "上面中間", "右上"], ["左邊中間", "正中間", "右邊中間"], ["左下", "下面中間", "右下"]];
const ZONE_GRID_FLOOR = [["前左", "前面中間", "前右"], ["左邊中間", "正中間", "右邊中間"], ["後左", "後面中間", "後右"]];

// ①…⑳，超過 20 用「(21)」。
function zoneNumberText(number) { return number >= 1 && number <= 20 ? String.fromCodePoint(0x245f + number) : `(${number})`; }

// 材料庫那一筆是不是門、窗、地毯（分組照材料庫那邊的規則：門窗是背後是空氣的那一組，地毯是地板組裡代號含 carpet 的）；不是回 null。
function zoneSpecialName(entry) {
  if (!entry?.placement) return null;
  if (entry.placement.group === "door_window") {
    if (entry.label.includes("門")) return "門";
    if (entry.label.includes("窗")) return "窗";
  }
  if (entry.placement.group === "floor" && entry.id.includes("carpet")) return "地毯";
  return null;
}

/* 一面牆每一區的號碼與名字。face：
   surface: "wall" | "floor" | "ceiling"；role（牆才有）："front" | "back" | "side" | null（牆名定不出前後）；
   width、height：畫面寬高（公尺）；tolerance：同一條邊的容差（公尺）；
   edges：{left, right, top, bottom} 四邊隔壁那一面的名字；front、left（地板天花才用）：前牆、左牆在畫面哪一邊，定不出是 null；
   zones：[{key, drawn, whole, box: [a0, a1, b0, b1], entry}]（entry 是這一區現在的材料對到的材料庫那一筆，對不到是 null）；
   speakers：[{name, at: [a, b]}]；primary：[a, b] 或 null。
   回 Map：區代號 → {number, numberText, name}。 */
function nameZones(face) {
  const tol = face.tolerance;
  const near = (x, y) => Math.abs(x - y) <= tol;
  const order = (left, right) => {
    if (!near(left.box[3], right.box[3])) return right.box[3] - left.box[3];
    return left.box[0] - right.box[0];
  };
  const numbered = [...face.zones.filter(zone => zone.drawn).sort(order), ...face.zones.filter(zone => !zone.drawn).sort(order)];
  const covers = (box, [a, b]) => a >= box[0] - tol && a <= box[1] + tol && b >= box[2] - tol && b <= box[3] + tol;
  // 長邊／短邊剛好 3 倍（5 公分格上常見）照容差算「條」，不跟著浮點誤差一下條一下塊。
  const shape = ([a0, a1, b0, b1]) => {
    const sides = [a1 - a0, b1 - b0].sort((x, y) => y - x);
    return sides[0] >= ZONE_STRIP_RATIO * sides[1] - tol ? "條" : "塊";
  };
  // 畫的框四邊都貼到牆邊，跟「整面一區」同一個名字。
  const fills = ([a0, a1, b0, b1]) => near(a0, 0) && near(a1, face.width) && near(b0, 0) && near(b1, face.height);
  const speakersHere = face.surface !== "wall" || face.role === "front";
  const speakerWhere = {wall: "後面", floor: "底下", ceiling: "上面"}[face.surface];
  const primaryWhere = face.surface === "floor" ? "主位底下" : face.surface === "ceiling" ? "主位上面"
    : face.role === "front" ? "主位正前方" : face.role === "back" ? "主位正後方" : null;
  // ⓪～② 的主角（「A 和 B 中間那條」也用它）：門／窗／地毯、蓋住的喇叭、主位；都沒有回 null。
  const subject = zone => {
    const special = zoneSpecialName(zone.entry);
    if (special) return {kind: "special", words: special, name: special};
    const speakers = speakersHere ? face.speakers.filter(speaker => covers(zone.box, speaker.at)) : [];
    if (speakers.length) {
      const words = speakers.length === 1 ? speakers[0].name
        : speakers.length === face.speakers.length && speakers.length === 2 ? "兩支喇叭"
        : speakers.length === 2 ? `${speakers[0].name}和${speakers[1].name}` : `${speakers.length} 支喇叭`;
      return {kind: "speaker", count: speakers.length, words, name: `${words}${speakerWhere}那${shape(zone.box)}`};
    }
    if (primaryWhere && face.primary && covers(zone.box, face.primary)) {
      return {kind: "primary", words: "主位", name: `${primaryWhere}那${shape(zone.box)}`};
    }
    return null;
  };
  const subjects = new Map(face.zones.map(zone => [zone.key, subject(zone)]));
  // 兩側夾著的、有主角的那一塊（重疊最長的那一塊）：side 是 "left"／"right"／"below"／"above"。
  const neighbour = (zone, side) => {
    const [a0, a1, b0, b1] = zone.box;
    let best = null, length = tol;
    for (const other of face.zones) {
      if (other === zone || !subjects.get(other.key)) continue;
      const [c0, c1, d0, d1] = other.box;
      const touching = {left: near(c1, a0), right: near(c0, a1), below: near(d1, b0), above: near(d0, b1)}[side];
      const overlap = side === "left" || side === "right" ? Math.min(b1, d1) - Math.max(b0, d0) : Math.min(a1, c1) - Math.max(a0, c0);
      if (touching && overlap > length) { best = other; length = overlap; }
    }
    return best && subjects.get(best.key);
  };
  const between = (one, two) => {
    if (!one || !two) return null;
    if (one.kind === "speaker" && two.kind === "speaker" && one.count === 1 && two.count === 1 && face.speakers.length === 2
        && one.words !== two.words) return "兩支喇叭";
    return one.words === two.words ? `兩塊${one.words}` : `${one.words}和${two.words}`;
  };
  // ④ 長條貼邊、夾在兩塊有名字的中間、橫跨或直跨整面不貼邊；都不是回 null（交給三等分）。
  const strip = zone => {
    const [a0, a1, b0, b1] = zone.box;
    const wide = near(a0, 0) && near(a1, face.width), tall = near(b0, 0) && near(b1, face.height);
    const long = shape(zone.box) === "條";
    const across = wide || (!tall && long && a1 - a0 > b1 - b0);
    const upright = !across && (tall || (long && b1 - b0 > a1 - a0));
    const piece = `那${shape(zone.box)}`;
    if (across && near(b1, face.height)) return `靠${face.edges.top}${piece}`;
    if (across && near(b0, 0)) return `靠${face.edges.bottom}${piece}`;
    if (upright && near(a0, 0)) return `靠${face.edges.left}${piece}`;
    if (upright && near(a1, face.width)) return `靠${face.edges.right}${piece}`;
    // 夾在中間：高的先看左右兩旁、扁的先看上下（兩支喇叭那兩塊中間的直條、門上門下夾著的橫條）。
    const sideways = between(neighbour(zone, "left"), neighbour(zone, "right"));
    const stacked = between(neighbour(zone, "above"), neighbour(zone, "below"));
    const middle = b1 - b0 > a1 - a0 ? sideways ?? stacked : stacked ?? sideways;
    if (middle) return `${middle}中間${piece}`;
    return wide || tall ? `中間${piece}` : null;
  };
  // 三等分：先把中心換成「離前牆／離左牆」的比例（地板天花定得出時），不然照畫面（離上緣、離左緣）。
  const third = value => (value < 1 / 3 ? 0 : value > 2 / 3 ? 2 : 1);
  const grid = zone => {
    const [a0, a1, b0, b1] = zone.box;
    const across = (a0 + a1) / 2 / face.width, up = (b0 + b1) / 2 / face.height;
    const fromEdge = {left: across, right: 1 - across, bottom: up, top: 1 - up};
    if (face.surface !== "wall" && face.front && face.left) {
      return `${ZONE_GRID_FLOOR[third(fromEdge[face.front])][third(fromEdge[face.left])]}那${shape(zone.box)}`;
    }
    return `${ZONE_GRID_WALL[third(fromEdge.top)][third(fromEdge.left)]}那${shape(zone.box)}`;
  };
  const named = new Map();
  numbered.forEach((zone, index) => {
    const number = index + 1;
    const name = zone.whole || fills(zone.box) ? "整面" : subjects.get(zone.key)?.name ?? strip(zone) ?? grid(zone);
    named.set(zone.key, {number, numberText: zoneNumberText(number), name});
  });
  return named;
}
