//! Entity graph and per-language "cards" built from the fixture export (`export/fixtures/*.json`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use kb_core::i18n::{LANGS, Translations};
use kb_core::store::{EdgeRow, EntityRow, NewDoc};
use kb_core::text::html_to_markdown;
use serde_json::{Value, json};

pub struct Output {
    pub entities: Vec<EntityRow>,
    pub names: Vec<(String, String, String)>,
    pub edges: Vec<EdgeRow>,
    pub docs: Vec<NewDoc>,
}

struct Fixtures {
    items: Value,
    actions: Value,
    buildings: Value,
    recipes: Value,
    ruins: Value,
    groups: Value,
    properties: Value,
    categories: Value,
    professions: Value,
    statuses: Value,
    roles: Value,
    hero_skills: Value,
    towns: Value,
    pictos: Value,
    home_levels: Value,
    home_upgrades: Value,
}

fn load(export: &Path, name: &str) -> Result<Value> {
    let path = export.join("fixtures").join(format!("{name}.json"));
    if !path.exists() {
        return Ok(Value::Null);
    }
    let raw = std::fs::read_to_string(&path)?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

/// Iterates an object as (key, value), or an array as (index, value).
fn entries(v: &Value) -> Vec<(String, &Value)> {
    match v {
        Value::Object(m) => m.iter().map(|(k, v)| (k.clone(), v)).collect(),
        Value::Array(a) => a.iter().enumerate().map(|(i, v)| (i.to_string(), v)).collect(),
        _ => vec![],
    }
}

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

fn num(v: &Value, key: &str) -> Option<f64> {
    match v.get(key)? {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn flag(v: &Value, key: &str) -> bool {
    num(v, key).is_some_and(|n| n != 0.0)
}

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

fn fmt_num(n: f64) -> String {
    if n.fract() == 0.0 { format!("{}", n as i64) } else { format!("{n}") }
}

fn pct(part: f64, total: f64) -> String {
    if total <= 0.0 { return "?".into() }
    let p = part / total * 100.0;
    if p >= 10.0 { format!("{p:.0}%") } else { format!("{p:.1}%") }
}

/// Strings or [key, weight] pairs, as used by recipe outputs. Returns (item key, weight).
fn weighted_list(v: &Value) -> Vec<(String, f64)> {
    match v {
        Value::String(s) => vec![(s.clone(), 1.0)],
        Value::Array(a) => a
            .iter()
            .filter_map(|e| match e {
                Value::String(s) => Some((s.clone(), 1.0)),
                Value::Array(pair) => Some((pair.first()?.as_str()?.to_string(), pair.get(1).and_then(Value::as_f64).unwrap_or(1.0))),
                _ => None,
            })
            .collect(),
        Value::Object(m) => m.iter().map(|(k, w)| (k.clone(), w.as_f64().unwrap_or(1.0))).collect(),
        _ => vec![],
    }
}

fn recipe_type(t: i64) -> &'static str {
    match t {
        1 => "Workshop",
        2 => "Workshop (shaman only)",
        3 => "Workshop (technician only)",
        11 => "Manual, outside the town",
        12 => "Manual, inside the town",
        13 => "Manual, anywhere",
        _ => "Recipe",
    }
}

/// Short form of an action atom: `StatusEffect(pointType=1, statusTo=hasdrunk)`.
fn describe_atom(atom: &Value) -> String {
    let class = s(atom, "atom").rsplit('\\').next().unwrap_or("Atom");
    let mut args = Vec::new();
    if let Some(Value::Object(payload)) = atom.get("payload") {
        for (k, v) in payload {
            match v {
                Value::Null => {}
                Value::Bool(false) => {}
                Value::String(s) => args.push(format!("{k}={s}")),
                other => args.push(format!("{k}={other}")),
            }
        }
    }
    if args.is_empty() { class.to_string() } else { format!("{class}({})", args.join(", ")) }
}

struct Ctx<'a> {
    f: &'a Fixtures,
    tr: &'a Translations,
}

impl Ctx<'_> {
    fn label(&self, lang: &str, domains: &[&str], de: &str) -> String {
        html_to_markdown(&self.tr.tr(lang, domains, de))
    }

    fn item_name(&self, lang: &str, key: &str) -> String {
        match self.f.items.get(key) {
            Some(item) => self.label(lang, &["items"], s(item, "label")),
            None => key.to_string(),
        }
    }

    fn building_name(&self, lang: &str, key: &str) -> String {
        match self.f.buildings.get(key) {
            Some(b) => self.label(lang, &["buildings"], s(b, "label")),
            None => key.to_string(),
        }
    }

    fn item_list(&self, lang: &str, v: &Value) -> String {
        weighted_list(v)
            .iter()
            .map(|(k, n)| if *n == 1.0 { self.item_name(lang, k) } else { format!("{}× {}", fmt_num(*n), self.item_name(lang, k)) })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn action(&self, id: &str) -> Option<&Value> {
        self.f.actions.get("actions")?.get(id)
    }

    fn describe_requirement(&self, lang: &str, id: &str) -> String {
        let Some(req) = self.f.actions.get("meta_requirements").and_then(|m| m.get(id)) else { return id.to_string() };
        let atoms: Vec<String> = req.get("atomList").map(entries).unwrap_or_default().iter().map(|(_, a)| describe_atom(a)).collect();
        let mut out = id.to_string();
        if !atoms.is_empty() {
            let _ = write!(out, " [{}]", atoms.join("; "));
        }
        let text = s(req, "text");
        if !text.is_empty() {
            let _ = write!(out, " — \"{}\"", self.label(lang, &["items", "game"], text).replace('\n', " "));
        }
        out
    }

    fn describe_result(&self, v: &Value, depth: usize) -> String {
        let id = match v {
            Value::String(id) => id.as_str(),
            other => return other.to_string(),
        };
        if depth < 2
            && let Some(group) = self.f.actions.get("results").and_then(|r| r.get("group")).and_then(|g| g.get(id))
        {
            let options: Vec<(Vec<&Value>, f64)> = entries(group)
                .into_iter()
                .filter_map(|(_, opt)| {
                    let arr = opt.as_array()?;
                    let results: Vec<&Value> = arr.first()?.as_array()?.iter().collect();
                    Some((results, arr.get(1).and_then(Value::as_f64).unwrap_or(1.0)))
                })
                .collect();
            let total: f64 = options.iter().map(|o| o.1).sum();
            let parts: Vec<String> = options
                .iter()
                .map(|(results, w)| {
                    let inner: Vec<String> = results.iter().map(|r| self.describe_result(r, depth + 1)).collect();
                    format!("{} → {}", pct(*w, total), inner.join(" + "))
                })
                .collect();
            return format!("{id}: one of {{ {} }}", parts.join(" | "));
        }
        match self.f.actions.get("meta_results").and_then(|m| m.get(id)) {
            Some(res) => {
                let atoms: Vec<String> = res.get("atomList").map(entries).unwrap_or_default().iter().map(|(_, a)| describe_atom(a)).collect();
                if atoms.is_empty() { id.to_string() } else { format!("{id} [{}]", atoms.join("; ")) }
            }
            None => id.to_string(),
        }
    }

    /// (action, item it is used with) for a job: actions gated on the job, then the actions of its
    /// starting items. Players know these as the job's abilities (the Scavenger's zone excavation
    /// is an action of the Small Spade), so the job card must list them.
    fn job_actions(&self, job: &str, profession: &Value) -> Vec<(String, Option<String>)> {
        let job_flag = format!("job_{job}");
        let gating: Vec<String> = self
            .f
            .actions
            .get("meta_requirements")
            .map(entries)
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, req)| {
                req.get("atomList").map(entries).unwrap_or_default().iter().any(|(_, a)| a.get("payload").and_then(|p| p.get(&job_flag)) == Some(&Value::Bool(true)))
            })
            .map(|(id, _)| id)
            .collect();
        let item_actions = self.f.actions.get("items");
        let item_with = |action: &str| -> Option<String> {
            entries(item_actions?).into_iter().find(|(_, list)| list.as_array().is_some_and(|l| l.iter().any(|a| a.as_str() == Some(action)))).map(|(item, _)| item)
        };

        let mut out: Vec<(String, Option<String>)> = Vec::new();
        for (id, action) in self.f.actions.get("actions").map(entries).unwrap_or_default() {
            let gated = action.get("meta").map(entries).unwrap_or_default().iter().any(|(_, r)| r.as_str().is_some_and(|r| gating.iter().any(|g| g == r)));
            if gated {
                let item = item_with(&id);
                out.push((id, item));
            }
        }
        for (item, _) in weighted_list(profession.get("items").unwrap_or(&Value::Null)) {
            let actions = item_actions.and_then(|m| m.get(&item)).and_then(Value::as_array).cloned().unwrap_or_default();
            for action in actions.iter().filter_map(Value::as_str) {
                if !out.iter().any(|(a, _)| a == action) {
                    out.push((action.to_string(), Some(item.clone())));
                }
            }
        }
        out
    }

    fn describe_action(&self, lang: &str, id: &str, out: &mut String) {
        let Some(a) = self.action(id) else {
            let _ = writeln!(out, "- `{id}`");
            return;
        };
        let label = self.label(lang, &["items", "game"], s(a, "label"));
        if label.is_empty() {
            let _ = writeln!(out, "- `{id}`");
        } else {
            let _ = writeln!(out, "- **{label}** (`{id}`)");
        }
        let reqs: Vec<String> = a.get("meta").map(entries).unwrap_or_default().iter().filter_map(|(_, r)| r.as_str()).map(|r| self.describe_requirement(lang, r)).collect();
        if !reqs.is_empty() {
            let _ = writeln!(out, "  - Requires: {}", reqs.join("; "));
        }
        let results: Vec<String> = a.get("result").map(entries).unwrap_or_default().iter().map(|(_, r)| self.describe_result(r, 0)).collect();
        if !results.is_empty() {
            let _ = writeln!(out, "  - Effects: {}", results.join("; "));
        }
        let message = s(a, "message");
        if !message.is_empty() {
            let _ = writeln!(out, "  - Message: \"{}\"", self.label(lang, &["items", "game"], message).replace('\n', " "));
        }
    }
}

fn also_known_as(names: &BTreeMap<&str, String>, lang: &str) -> String {
    let mut others: Vec<&String> = names.iter().filter(|(l, _)| **l != lang).map(|(_, n)| n).collect();
    others.dedup();
    others.into_iter().filter(|n| Some(*n) != names.get(lang)).cloned().collect::<Vec<_>>().join(" / ")
}

pub fn build(export: &Path, tr: &Translations, langs: &[String], git_sha: Option<&str>) -> Result<Output> {
    let f = Fixtures {
        items: load(export, "Item")?,
        actions: load(export, "Action")?,
        buildings: load(export, "Building")?,
        recipes: load(export, "Recipe")?,
        ruins: load(export, "Ruin")?,
        groups: load(export, "ItemGroup")?,
        properties: load(export, "ItemProperty")?,
        categories: load(export, "ItemCategory")?,
        professions: load(export, "CitizenProfession")?,
        statuses: load(export, "CitizenStatus")?,
        roles: load(export, "CitizenRole")?,
        hero_skills: load(export, "HeroSkill")?,
        towns: load(export, "Town")?,
        pictos: load(export, "Picto")?,
        home_levels: load(export, "CitizenHomeLevel")?,
        home_upgrades: load(export, "CitizenHomeUpgrade")?,
    };
    let ctx = Ctx { f: &f, tr };
    let mut out = Output { entities: vec![], names: vec![], edges: vec![], docs: vec![] };
    let edge = |src: String, rel: &str, dst: String, weight: Option<f64>| EdgeRow { src, rel: rel.to_string(), dst, weight };

    // ---- Graph: entities, names, edges -------------------------------------------------
    let simple_kinds: [(&str, &Value, &[&str], &str); 7] = [
        ("building", &f.buildings, &["buildings"], "label"),
        ("ruin", &f.ruins, &["game"], "label"),
        ("profession", &f.professions, &["game"], "label"),
        ("status", &f.statuses, &["game"], "label"),
        ("role", &f.roles, &["game"], "label"),
        ("heroskill", &f.hero_skills, &["game"], "title"),
        ("town", &f.towns, &["game"], "label"),
    ];
    for (key, item) in entries(&f.items) {
        let id = format!("item:{key}");
        out.entities.push(EntityRow { id: id.clone(), kind: "item".into(), data: item.clone() });
        for lang in LANGS {
            out.names.push((id.clone(), lang.into(), ctx.item_name(lang, &key)));
        }
        for (_, action) in f.actions.get("items").and_then(|m| m.get(&key)).map(entries).unwrap_or_default() {
            if let Some(a) = action.as_str() {
                out.edges.push(edge(id.clone(), "has_action", format!("action:{a}"), None));
            }
        }
    }
    for (kind, data, domains, field) in simple_kinds {
        for (key, v) in entries(data) {
            let id = format!("{kind}:{key}");
            out.entities.push(EntityRow { id: id.clone(), kind: kind.into(), data: v.clone() });
            for lang in LANGS {
                out.names.push((id.clone(), lang.into(), ctx.label(lang, domains, s(v, field))));
            }
        }
    }
    for (key, a) in f.actions.get("actions").map(entries).unwrap_or_default() {
        out.entities.push(EntityRow { id: format!("action:{key}"), kind: "action".into(), data: a.clone() });
    }
    for (key, r) in entries(&f.recipes) {
        let id = format!("recipe:{key}");
        out.entities.push(EntityRow { id: id.clone(), kind: "recipe".into(), data: r.clone() });
        for (input, _) in weighted_list(r.get("in").unwrap_or(&Value::Null)) {
            out.edges.push(edge(format!("item:{input}"), "input_of", id.clone(), None));
        }
        for (output, w) in weighted_list(r.get("out").unwrap_or(&Value::Null)) {
            out.edges.push(edge(id.clone(), "produces", format!("item:{output}"), Some(w)));
        }
    }
    for (key, b) in entries(&f.buildings) {
        let id = format!("building:{key}");
        let parent = s(b, "parentBuilding");
        if !parent.is_empty() {
            out.edges.push(edge(id.clone(), "child_of", format!("building:{parent}"), None));
        }
        for (item, n) in weighted_list(b.get("resources").unwrap_or(&Value::Null)) {
            out.edges.push(edge(id.clone(), "requires", format!("item:{item}"), Some(n)));
        }
    }
    for (key, r) in entries(&f.ruins) {
        for (item, w) in weighted_list(r.get("drops").unwrap_or(&Value::Null)) {
            out.edges.push(edge(format!("ruin:{key}"), "drops", format!("item:{item}"), Some(w)));
        }
    }
    for (group, items) in entries(&f.groups) {
        out.entities.push(EntityRow { id: format!("itemgroup:{group}"), kind: "itemgroup".into(), data: items.clone() });
        for (item, w) in weighted_list(items) {
            out.edges.push(edge(format!("itemgroup:{group}"), "drops", format!("item:{item}"), Some(w)));
        }
    }

    // ---- Cards ----------------------------------------------------------------------------
    let doc = |entity_id: &str, lang: &str, title: String, content: String| NewDoc {
        key: format!("card:{entity_id}:{lang}"),
        kind: "entity_card".into(),
        audience: "both".into(),
        lang: Some(lang.to_string()),
        title,
        entity_id: Some(entity_id.to_string()),
        git_sha: git_sha.map(str::to_string),
        content,
        meta: json!({ "entity_kind": entity_id.split(':').next().unwrap_or("") }),
        ..Default::default()
    };

    for lang in langs.iter().map(String::as_str) {
        for (key, item) in entries(&f.items) {
            out.docs.push(item_card(&ctx, lang, &key, item, &doc));
        }
        for (key, b) in entries(&f.buildings) {
            out.docs.push(building_card(&ctx, lang, &key, b, &doc));
        }
        for (key, r) in entries(&f.ruins) {
            out.docs.push(ruin_card(&ctx, lang, &key, r, &doc));
        }
        for (key, p) in entries(&f.professions) {
            let id = format!("profession:{key}");
            let title = ctx.label(lang, &["game"], s(p, "label"));
            let mut c = format!("Job (profession) `{key}` · Hero job: {}\n\n", yes_no(flag(p, "hero")));
            let desc = s(p, "desc");
            if !desc.is_empty() {
                let _ = writeln!(c, "{}\n", ctx.label(lang, &["game"], desc));
            }
            let _ = writeln!(c, "Starting items: {}", ctx.item_list(lang, p.get("items").unwrap_or(&Value::Null)));
            if let Some(alt) = p.get("items_alt") {
                let _ = writeln!(c, "Alternative starting items: {}", ctx.item_list(lang, alt));
            }
            if let Some(bonus) = num(p, "dig_bonus") {
                let _ = writeln!(c, "Digging bonus: +{}%", fmt_num(bonus * 100.0));
            }
            let actions = ctx.job_actions(&key, p);
            if !actions.is_empty() {
                let _ = writeln!(c, "
## Job abilities (actions only this job can use, or that come with its starting items)");
                let mut heading = None;
                for (action, item) in &actions {
                    if heading != Some(item) {
                        heading = Some(item);
                        let _ = match item {
                            Some(item) => writeln!(c, "With {}:", ctx.item_name(lang, item)),
                            None => writeln!(c, "Job-only action:"),
                        };
                    }
                    ctx.describe_action(lang, action, &mut c);
                }
            }
            out.docs.push(doc(&id, lang, title, c));
        }
        for (key, st) in entries(&f.statuses) {
            let id = format!("status:{key}");
            let title = ctx.label(lang, &["game"], s(st, "label"));
            let mut c = format!("Citizen status `{key}` · Volatile (cleared at night): {}\n\n", yes_no(flag(st, "volatile")));
            let _ = writeln!(c, "{}", ctx.label(lang, &["game"], s(st, "description")));
            let extra: Vec<String> = entries(st)
                .into_iter()
                .filter(|(k, _)| !["name", "label", "description", "volatile", "icon"].contains(&k.as_str()))
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            if !extra.is_empty() {
                let _ = writeln!(c, "\nProperties: {}", extra.join(", "));
            }
            out.docs.push(doc(&id, lang, title, c));
        }
        for (key, r) in entries(&f.roles) {
            let id = format!("role:{key}");
            let title = ctx.label(lang, &["game"], s(r, "label"));
            let c = format!(
                "Citizen role `{key}` · Elected by vote: {} · Secret: {}\nHelp page: {}\n",
                yes_no(flag(r, "vote")),
                yes_no(flag(r, "secret")),
                s(r, "help")
            );
            out.docs.push(doc(&id, lang, title, c));
        }
        for (key, h) in entries(&f.hero_skills) {
            let id = format!("heroskill:{key}");
            let title = ctx.label(lang, &["game"], s(h, "title"));
            let mut c = format!("Hero skill `{key}`");
            if let Some(days) = num(h, "unlockAt") {
                let _ = write!(c, " · Unlocked at {} hero days", fmt_num(days));
            }
            if flag(h, "legacy") {
                c.push_str(" · Legacy skill");
            }
            let _ = writeln!(c, "\n\n{}", ctx.label(lang, &["game"], s(h, "description")));
            for (prop, v) in h.get("citizenProperties").map(entries).unwrap_or_default() {
                let _ = writeln!(c, "- Citizen property `{prop}` = {v}");
            }
            out.docs.push(doc(&id, lang, title, c));
        }
        for (key, t) in entries(&f.towns) {
            let id = format!("town:{key}");
            let title = ctx.label(lang, &["game"], s(t, "label"));
            let c = format!(
                "Town type `{key}` · Counts for season ranking: {}\n\n{}\n\nThe exact rules of this town type are in the \"Town rules\" documents for `{key}`.\n",
                if t.get("ranked").is_some_and(|r| r.is_array() || r.as_bool() == Some(true)) { "yes" } else { "no" },
                ctx.label(lang, &["game"], s(t, "help")),
            );
            out.docs.push(doc(&id, lang, title, c));
        }

        // Lists without stable keys become one document each
        let mut pictos = String::from("Pictos (rewards) that citizens can earn, with their description.\n\n");
        for (_, p) in entries(&f.pictos) {
            let _ = writeln!(
                pictos,
                "- **{}** (`{}`{}): {}",
                ctx.label(lang, &["game"], s(p, "label")),
                s(p, "icon"),
                if flag(p, "rare") { ", rare" } else { "" },
                ctx.label(lang, &["game"], s(p, "description")).replace('\n', " ")
            );
        }
        out.docs.push(NewDoc { key: format!("card:list:pictos:{lang}"), ..doc("list:pictos", lang, "Pictos".into(), pictos) });

        let mut homes = String::from("Home levels: defense, action point cost (with/without Urbanism), and resources needed to build each level.\n\n");
        for (i, h) in entries(&f.home_levels) {
            let _ = writeln!(
                homes,
                "- Level {i}: **{}** · defense {} · {} AP ({} AP with Urbanism) · resources: {}{}",
                ctx.label(lang, &["buildings", "game"], s(h, "label")),
                fmt_num(num(h, "def").unwrap_or(0.0)),
                fmt_num(num(h, "ap").unwrap_or(0.0)),
                fmt_num(num(h, "ap_urbanism").unwrap_or(0.0)),
                match ctx.item_list(lang, h.get("resources").unwrap_or(&Value::Null)) { s if s.is_empty() => "none".into(), s => s },
                if s(h, "building").is_empty() { String::new() } else { format!(" · requires building {}", ctx.building_name(lang, s(h, "building"))) },
            );
        }
        let _ = writeln!(homes, "\nHome upgrades (bought per level: [AP cost, resources]):\n");
        for (_, u) in entries(&f.home_upgrades) {
            let _ = writeln!(
                homes,
                "- **{}** (`{}`): {} Levels: {}",
                ctx.label(lang, &["buildings", "game"], s(u, "label")),
                s(u, "name"),
                ctx.label(lang, &["buildings", "game"], s(u, "desc")).replace('\n', " "),
                u.get("levels").map(Value::to_string).unwrap_or_default()
            );
        }
        out.docs.push(NewDoc { key: format!("card:list:home:{lang}"), ..doc("list:home", lang, "Home levels and upgrades".into(), homes) });
    }

    Ok(out)
}

fn names_by_lang(f: impl Fn(&str) -> String) -> BTreeMap<&'static str, String> {
    LANGS.iter().map(|l| (*l, f(l))).collect()
}

fn item_card(ctx: &Ctx, lang: &str, key: &str, item: &Value, doc: &impl Fn(&str, &str, String, String) -> NewDoc) -> NewDoc {
    let f = ctx.f;
    let id = format!("item:{key}");
    let names = names_by_lang(|l| ctx.item_name(l, key));
    let title = names[lang].clone();
    let mut c = String::new();

    let category = entries(&f.categories)
        .into_iter()
        .find(|(_, cat)| s(cat, "name") == s(item, "category"))
        .map(|(_, cat)| ctx.label(lang, &["items"], s(cat, "label")))
        .unwrap_or_else(|| s(item, "category").to_string());
    let _ = writeln!(c, "Item `{key}` · Other names: {}", also_known_as(&names, lang));
    let _ = writeln!(
        c,
        "Category: {category} · Heavy: {} · Decoration points: {} · Night watch defense: {}",
        yes_no(flag(item, "heavy")),
        fmt_num(num(item, "deco").unwrap_or(0.0)),
        fmt_num(num(item, "watchpoint").unwrap_or(0.0))
    );
    let props: Vec<&str> = f.properties.get(key).map(entries).unwrap_or_default().iter().filter_map(|(_, p)| p.as_str()).collect();
    if !props.is_empty() {
        let _ = writeln!(c, "Properties: {}", props.join(", "));
    }
    let desc = s(item, "description");
    if !desc.is_empty() {
        let _ = writeln!(c, "\n{}", ctx.label(lang, &["items"], desc));
    }

    let actions: Vec<String> = f.actions.get("items").and_then(|m| m.get(key)).map(entries).unwrap_or_default().iter().filter_map(|(_, a)| a.as_str().map(str::to_string)).collect();
    if !actions.is_empty() {
        let _ = writeln!(c, "\n## Actions");
        for a in &actions {
            ctx.describe_action(lang, a, &mut c);
        }
    }
    for (section, field) in [("Night watch", "items_nw"), ("Catapult", "items_cata")] {
        if let Some(v) = f.actions.get(field).and_then(|m| m.get(key)) {
            let _ = writeln!(c, "\n## {section}");
            for (_, a) in entries(v).into_iter().chain(v.as_str().map(|_| (String::new(), v))) {
                if let Some(a) = a.as_str() {
                    ctx.describe_action(lang, a, &mut c);
                }
            }
        }
    }

    let mut obtained = Vec::new();
    let mut used = Vec::new();
    for (rid, r) in entries(&f.recipes) {
        let ins = weighted_list(r.get("in").unwrap_or(&Value::Null));
        let outs = weighted_list(r.get("out").unwrap_or(&Value::Null));
        let total: f64 = outs.iter().map(|o| o.1).sum();
        let kind = recipe_type(r.get("type").and_then(Value::as_i64).unwrap_or(0));
        let verb = match s(r, "action") {
            "" => String::new(),
            action => format!(" ({})", ctx.label(lang, &["items", "game"], action)),
        };
        if let Some((_, w)) = outs.iter().find(|(o, _)| o == key) {
            let chance = if outs.len() > 1 { format!(" ({} chance)", pct(*w, total)) } else { String::new() };
            obtained.push(format!("{kind} `{rid}`{verb} from: {}{chance}", ctx.item_list(lang, r.get("in").unwrap_or(&Value::Null))));
        }
        if ins.iter().any(|(i, _)| i == key) {
            let results: Vec<String> = outs.iter().map(|(o, w)| if outs.len() > 1 { format!("{} {}", ctx.item_name(lang, o), pct(*w, total)) } else { ctx.item_name(lang, o) }).collect();
            let with = ins.iter().filter(|(i, _)| i != key).map(|(i, _)| ctx.item_name(lang, i)).collect::<Vec<_>>();
            let with = if with.is_empty() { String::new() } else { format!(" together with {}", with.join(", ")) };
            used.push(format!("{kind} `{rid}`{verb}{with} → {}", results.join(", ")));
        }
    }
    if !obtained.is_empty() {
        let _ = writeln!(c, "\n## Obtained by transformation\n{}", obtained.iter().map(|l| format!("- {l}\n")).collect::<String>());
    }
    if !used.is_empty() {
        let _ = writeln!(c, "\n## Can be transformed\n{}", used.iter().map(|l| format!("- {l}\n")).collect::<String>());
    }

    let mut found = Vec::new();
    for (rkey, ruin) in entries(&f.ruins) {
        let drops = weighted_list(ruin.get("drops").unwrap_or(&Value::Null));
        let total: f64 = drops.iter().map(|d| d.1).sum();
        if let Some((_, w)) = drops.iter().find(|(i, _)| i == key) {
            found.push(format!("Ruin {} (`{rkey}`): {} of finds", ctx.label(lang, &["game"], s(ruin, "label")), pct(*w, total)));
        }
    }
    for (group, items) in entries(&f.groups) {
        let drops = weighted_list(items);
        let total: f64 = drops.iter().map(|d| d.1).sum();
        if let Some((_, w)) = drops.iter().find(|(i, _)| i == key) {
            found.push(format!("Item table `{group}` (e.g. digging in the desert): {} of drops", pct(*w, total)));
        }
    }
    if !found.is_empty() {
        let _ = writeln!(c, "\n## Where to find it\n{}", found.iter().map(|l| format!("- {l}\n")).collect::<String>());
    }

    let mut builds = Vec::new();
    for (bkey, b) in entries(&f.buildings) {
        if let Some((_, n)) = weighted_list(b.get("resources").unwrap_or(&Value::Null)).into_iter().find(|(i, _)| i == key) {
            builds.push(format!("{} ×{}", ctx.building_name(lang, &bkey), fmt_num(n)));
        }
    }
    if !builds.is_empty() {
        let _ = writeln!(c, "\n## Needed to build\n{}", builds.join(", "));
    }

    doc(&id, lang, title, c)
}

fn building_card(ctx: &Ctx, lang: &str, key: &str, b: &Value, doc: &impl Fn(&str, &str, String, String) -> NewDoc) -> NewDoc {
    let f = ctx.f;
    let id = format!("building:{key}");
    let names = names_by_lang(|l| ctx.building_name(l, key));
    let title = names[lang].clone();
    let mut c = String::new();
    let _ = writeln!(c, "Construction `{key}` · Other names: {}", also_known_as(&names, lang));
    let _ = writeln!(
        c,
        "Defense: {} · Action points: {} · Structure points: {} · Temporary: {} · Blueprint level: {}",
        fmt_num(num(b, "defense").unwrap_or(0.0)),
        fmt_num(num(b, "ap").unwrap_or(0.0)),
        num(b, "health").map(fmt_num).unwrap_or_else(|| "-".into()),
        yes_no(flag(b, "isTemporary")),
        fmt_num(num(b, "blueprintLevel").unwrap_or(0.0)),
    );
    if flag(b, "isImpervious") {
        let _ = writeln!(c, "Impervious: cannot be damaged by the attack.");
    }
    let parent = s(b, "parentBuilding");
    if !parent.is_empty() {
        let _ = writeln!(c, "Requires (parent construction): {}", ctx.building_name(lang, parent));
    }
    let children: Vec<String> = entries(&f.buildings).into_iter().filter(|(_, x)| s(x, "parentBuilding") == key).map(|(k, _)| ctx.building_name(lang, &k)).collect();
    if !children.is_empty() {
        let _ = writeln!(c, "Unlocks: {}", children.join(", "));
    }
    let desc = s(b, "description");
    if !desc.is_empty() {
        let _ = writeln!(c, "\n{}", ctx.label(lang, &["buildings"], desc));
    }
    let resources = ctx.item_list(lang, b.get("resources").unwrap_or(&Value::Null));
    let _ = writeln!(c, "\nResources: {}", if resources.is_empty() { "none".into() } else { resources });
    if flag(b, "hasHardMode") {
        if let Some(r) = b.get("hardResources") {
            let _ = writeln!(c, "Hard mode (Pandemonium): {} AP, resources: {}", fmt_num(num(b, "hardAp").unwrap_or(0.0)), ctx.item_list(lang, r));
        }
        if let Some(r) = b.get("easyResources") {
            let _ = writeln!(c, "Easy mode: {} AP, resources: {}", fmt_num(num(b, "easyAp").or(num(b, "ap")).unwrap_or(0.0)), ctx.item_list(lang, r));
        }
    }
    if let Some(level) = num(b, "voteLevel") {
        let _ = writeln!(c, "\n## Daily upgrades (up to level {})", fmt_num(level));
        let base = s(b, "baseVoteText");
        if !base.is_empty() {
            let _ = writeln!(c, "- Level 0: {}", ctx.label(lang, &["buildings"], base));
        }
        for (i, t) in b.get("upgradeTexts").map(entries).unwrap_or_default() {
            let lvl: usize = i.parse().unwrap_or(0) + 1;
            let _ = writeln!(c, "- Level {lvl}: {}", ctx.label(lang, &["buildings"], t.as_str().unwrap_or("")));
        }
    }
    doc(&id, lang, title, c)
}

fn ruin_card(ctx: &Ctx, lang: &str, key: &str, r: &Value, doc: &impl Fn(&str, &str, String, String) -> NewDoc) -> NewDoc {
    let id = format!("ruin:{key}");
    let names = names_by_lang(|l| ctx.label(l, &["game"], s(r, "label")));
    let title = names[lang].clone();
    let mut c = String::new();
    let _ = writeln!(c, "Ruin (building in the World Beyond) `{key}` · Other names: {}", also_known_as(&names, lang));
    let _ = writeln!(
        c,
        "Distance from town: {}–{} km · Camping bonus: {} · Spawn weight: {} · Chance to be empty when searched: {} · Capacity: {}",
        fmt_num(num(r, "min_dist").unwrap_or(0.0)),
        fmt_num(num(r, "max_dist").unwrap_or(0.0)),
        fmt_num(num(r, "camping").unwrap_or(0.0)),
        fmt_num(num(r, "chance").unwrap_or(0.0)),
        num(r, "empty").map(|e| format!("{}%", fmt_num(e * 100.0))).unwrap_or_else(|| "-".into()),
        fmt_num(num(r, "capacity").unwrap_or(0.0)),
    );
    let desc = s(r, "desc");
    if !desc.is_empty() {
        let _ = writeln!(c, "\n{}", ctx.label(lang, &["game"], desc));
    }
    let drops = weighted_list(r.get("drops").unwrap_or(&Value::Null));
    let total: f64 = drops.iter().map(|d| d.1).sum();
    if !drops.is_empty() {
        let _ = writeln!(c, "\n## Items found when searching");
        let mut sorted = drops.clone();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (item, w) in sorted {
            let _ = writeln!(c, "- {}: {}", ctx.item_name(lang, &item), pct(w, total));
        }
    }
    doc(&id, lang, title, c)
}
