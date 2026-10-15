from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Callable, cast

import httpx
import requests
from openai import OpenAI


def big_nhl_boy(user_statement: str, api_key: str) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, list[str]],
    dict[str, list[str]], list[Callable[[set[str], set[str]], bool]]
]:
    """Collect current ESPN NHL team rosters and return four preference-filtered pools
    plus callable, explicit AND/OR/NOT trade requirements. No trade generation
    or NHL contract/legal eligibility validation is performed.
    """
    BATCH_SIZE = 5
    ESPN_API = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
    REQUEST_DELAY = 0.15

    if not isinstance(user_statement, str) or not user_statement.strip():
        raise ValueError("user_statement must be a nonempty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a nonempty string")

    def normalize_name(value):
        return re.sub(r"\s+", " ", str(value)).strip()

    # ============================================================
    # STAGE 1 — COLLECT ALL 32 NHL TEAM ROSTERS FROM ESPN
    # ============================================================
    def collect_nhl_rosters() -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; NHLTradeEngine/1.0)",
            "Accept": "application/json",
        })

        def fetch(path: str, params: dict | None = None) -> dict:
            for attempt in range(3):
                try:
                    response = session.get(f"{ESPN_API}{path}", params=params, timeout=25)
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON object")
                    return payload
                except (requests.RequestException, ValueError) as exc:
                    if attempt == 2:
                        raise RuntimeError(f"ESPN NHL API request failed: {path}") from exc
                    time.sleep(attempt + 1)
            raise RuntimeError("Unreachable request state")

        raw_teams = fetch("/teams", {"limit": 100}).get("sports", [])
        team_rows = [team for sport in raw_teams
                     for league in sport.get("leagues", [])
                     for team in league.get("teams", [])]
        # The ESPN teams endpoint nests each record under its 'team' field.
        teams = sorted((row.get("team", row) for row in team_rows),
                       key=lambda team: team.get("displayName", ""))
        if len(teams) != 32:
            raise RuntimeError(f"Expected 32 NHL teams from ESPN; received {len(teams)}")

        league_players: dict[str, list[str]] = {}
        player_ids: dict[str, dict[str, str]] = {}

        for index, team in enumerate(teams, 1):
            name = team.get("displayName")
            team_id = team.get("id")
            if not name or not team_id:
                raise RuntimeError(f"Incomplete ESPN team entry: {team!r}")
            roster_data = fetch(f"/teams/{team_id}/roster")
            athletes = roster_data.get("athletes", [])
            # ESPN returns either a flat athletes array or groups containing
            # 'items' (forwards / defensemen / goalies), depending on endpoint.
            players: dict[str, str] = {}
            for group in athletes:
                entries = group.get("items", []) if isinstance(group, dict) and "items" in group else [group]
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    athlete = entry.get("athlete", entry)
                    if not isinstance(athlete, dict):
                        continue
                    player_name = athlete.get("fullName") or athlete.get("displayName")
                    player_id = athlete.get("id")
                    if player_name and player_id:
                        players[normalize_name(player_name)] = str(player_id)
            if not players:
                raise RuntimeError(f"Empty ESPN NHL roster for {name}; refusing incomplete data")
            league_players[name] = sorted(players)
            player_ids[name] = players
            print(f"[{index:02d}/32] {name}: {len(players)} players")
            time.sleep(REQUEST_DELAY)
        return league_players, player_ids

    league_players, player_ids = collect_nhl_rosters()
    client = OpenAI(api_key=api_key, http_client=cast(Any, httpx.Client()))

    # ============================================================
    # STAGE 2 — CURRENT PLAYER EVALUATIONS (5 TEAMS PER GPT BATCH)
    # ============================================================
    NHL_PROMPT_1 = 'Below is a list of current NHL players grouped by team.\n\n{batch_text}\n\nEvaluate EVERY player listed.\n\nFor each player, return:\n\n1. Attributes:\n- Up to 8 concise CURRENT attributes describing players physically, mentally,\n  stylistically, or by career stage.\n- Examples: age, young, veteran, height, size, strength, speed, acceleration, agility, lateral mobility, skating stride, endurance, durability, physicality, high motor, hockey IQ, leadership (only with evidence), playoff experience, injury history, consistency.\n\n2. Strengths:\n- Up to 5 meaningful CURRENT hockey strengths, where supported.\n- Possible areas: skating speed, acceleration, edgework, puck handling, stickhandling, puck protection, zone entries, controlled exits, passing, playmaking, vision, offensive-zone creation, shooting, wrist shots, slap shots, one-timers, finishing, net-front presence, screen-setting, deflections, forechecking, backchecking, puck retrieval, faceoff ability, board battles, checking, physicality, penalty killing, power-play production, transition play, defensive positioning, defensive-zone coverage, shot blocking, gap control, rush defense, stick checking, takeaways, limiting turnovers, defensive versatility, goaltending, save percentage, rebound control, positioning, lateral movement, high-danger save ability, puck tracking, puck handling as a goaltender, reliability.\n- Make strengths specific to a player\'s position and role; do not assign\n  positional skills to inapplicable players.\n\n3. Weaknesses:\n- Up to 5 meaningful CURRENT hockey weaknesses, drawing from applicable\n  categories above and each player\'s actual role.\n\nIMPORTANT RULES:\n- Be stringent. Evaluate CURRENT ability only, not historical peak or speculative potential.\n- Do NOT manufacture strengths or weaknesses to reach the maximum counts.\n- Distinguish reputation from current performance.\n- Only call someone clutch or a poor pressure performer with meaningful evidence.\n- Do not infer personality traits without reasonable public evidence.\n- Use precise hockey terminology so later code can filter players by roles,\n  attributes, strengths and weaknesses.\n- Evaluate EVERY player supplied without omission; use current information when necessary.\n- Do not move players between teams or invent players.\n\nReturn exactly the following format, for every team and player:\nTeam Name:\nPlayer Name: Attributes ["attribute 1", ...], Strengths ["strength 1", ...], Weaknesses ["weakness 1", ...]\nPlayer Name 2: Attributes [...], Strengths [...], Weaknesses [...]\nNext Team Name:\nPlayer Name: Attributes [...], Strengths [...], Weaknesses [...]\n\nReturn ONLY the requested team/player information, with no introductions,\nmethodology, citations, source lists, conclusions, or disclaimers.'

    teams = list(league_players.items())
    all_results: list[str] = []
    total_batches = (len(teams) + BATCH_SIZE - 1) // BATCH_SIZE
    for start_index in range(0, len(teams), BATCH_SIZE):
        batch_text = json.dumps(dict(teams[start_index:start_index + BATCH_SIZE]), ensure_ascii=False, indent=2)
        print(f"NHL player evaluation batch {(start_index // BATCH_SIZE) + 1}/{total_batches}")
        response = client.responses.create(
            model="gpt-5.6", tools=[{"type": "web_search"}],
            input=NHL_PROMPT_1.replace("{batch_text}", batch_text)
        )
        all_results.append(response.output_text.strip())

    # ============================================================
    # STAGE 3 — PARSE THE USER'S TRADE PREFERENCES USING GPT
    # ============================================================
    
    NHL_PROMPT_2 = 'You are interpreting an NHL trade request for a later player-pool filter.\nRead the user statement and extract the following, without inventing names:\n\nTeam Name: [user\'s NHL team]\nRequired Acquisition Players: [named players the user explicitly requires acquiring]\nPreferred Acquisition Players: [named players the user wants or is interested in but does not require]\nAcquisition Player Types: [roles, positions, skills, traits of acceptable acquisitions]\nPreferred Outgoing Players: [named players the user explicitly wants to trade away]\nOther Allowed Outgoing Players: [named players the user permits trading without preferring to move]\nUntouchable Players: [named players never to trade]\nUntouchable Player Types: [traits or roles of players never to trade]\nPlayers to Avoid Acquiring: [specific names]\nPlayer Types to Avoid Acquiring: [traits, weaknesses, positions, roles to exclude]\nTeams to Avoid: [teams the user refuses to deal with]\nRequired Trade Logic: [Python Boolean membership expressions, as quoted strings]\n\nExtract mandatory logical conditions ONLY when explicitly stated by the user.\nUse Python expressions with membership tests, and/or/not operators, parentheses,\nand the variables acquired and outgoing. Use full player names.\nExample:\nRequired Trade Logic: ["(\'Player A\' in acquired or \'Player B\' in acquired)", "not (\'Player A\' in acquired and \'Player B\' in acquired)", "(\'Player C\' not in outgoing or \'Player A\' in acquired)"]\n\n- "Either A or B, but not both" requires two expressions: A-or-B and not(A-and-B).\n- "Trade C only if acquiring A" means C not in outgoing or A in acquired.\n- "I must acquire both A and B" means A in acquired and B in acquired.\n- Use acquired ONLY for players RECEIVED by the user\'s team, and outgoing\n  ONLY for players SENT by the user\'s team.\n- The five-output system consists of four separate player dictionaries plus\n  this logic list; the list can relate players across any of the four dictionaries.\n- Do NOT infer mandatory logic for merely preferred/allowable players.\n- Do NOT introduce conditions for every player in a pool.\n- Do NOT invent position-level Boolean rules, lambda expressions, function calls,\n  or variable names other than acquired and outgoing.\n- If there is no explicit mandatory logical condition, use [].\n- Keep AND vs OR vs NOT faithful to the user\'s wording.\n\nDistinguish REQUIRED from merely PREFERRED acquisitions. A name being mentioned\nis not automatically mandatory. Likewise, "willing to part with" is permission,\nnot a desire to trade that player. If a category is absent, put [].\nDo not invent NHL rules, roster status, contract eligibility, or trade legality.\n\nUSER STATEMENT:\n{user_statement}\n\nReturn only the labeled fields above, in the order shown. No Markdown or explanations.'

    demands = client.responses.create(
        model="gpt-5.6",
        input=NHL_PROMPT_2.replace("{user_statement}", user_statement),
    )

    print("\nPARSED USER PREFERENCES:\n", demands.output_text)
    # Parse the required trade-logic list separately from the preference labels.
    logic_match = re.search(r"(?im)^Required Trade Logic:\s*(\[[^\n]*\])\s*$", demands.output_text)
    if logic_match is None:
        raise ValueError("GPT omitted Required Trade Logic list")
    raw_logic = ast.literal_eval(logic_match.group(1))
    if not isinstance(raw_logic, list) or not all(isinstance(x, str) for x in raw_logic):
        raise ValueError("Required Trade Logic must be a list of Python expressions")

    # Accept only name membership and AND/OR/NOT. Do not execute GPT-generated code.
    # These predicates are evaluated later, once for each proposed trade.
    def make_trade_rule(expression):
        tree = ast.parse(expression, mode="eval")

        def check_syntax(node):
            if isinstance(node, ast.Expression):
                return check_syntax(node.body)
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                if len(node.values) < 2:
                    raise ValueError("Boolean group requires at least two members")
                for child in node.values:
                    check_syntax(child)
                return
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return check_syntax(node.operand)
            if (isinstance(node, ast.Compare) and len(node.ops) == 1
                    and isinstance(node.ops[0], (ast.In, ast.NotIn))
                    and isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and isinstance(node.comparators[0], ast.Name)
                    and node.comparators[0].id in {"acquired", "outgoing"}):
                return
            raise ValueError(f"Unsupported trade logic expression: {expression!r}")

        check_syntax(tree)

        def evaluate(node, acquired, outgoing):
            if isinstance(node, ast.Expression):
                return evaluate(node.body, acquired, outgoing)
            if isinstance(node, ast.BoolOp):
                results = (evaluate(n, acquired, outgoing) for n in node.values)
                return all(results) if isinstance(node.op, ast.And) else any(results)
            if isinstance(node, ast.UnaryOp):
                return not evaluate(node.operand, acquired, outgoing)
            selected = acquired if node.comparators[0].id == "acquired" else outgoing
            present = node.left.value in selected
            return present if isinstance(node.ops[0], ast.In) else not present

        def rule(acquired, outgoing):
            return evaluate(tree, set(acquired), set(outgoing))

        rule.expression = expression
        return rule

    # Verify all referenced players against the current NHL rosters. Keep the
    # resulting predicates intact even when an eligibility/preference filter
    # excludes a named player: missing that player will cause the relevant
    # requirement to fail when checking an actual trade.
    known_names = {normalize_name(p).casefold(): p for roster in league_players.values() for p in roster}
    canonical_logic = []
    for expression in raw_logic:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                original = node.left.value
                actual = known_names.get(normalize_name(original).casefold())
                if actual is None:
                    raise ValueError(f"Required trade logic references player not on a scraped roster: {original!r}")
                node.left.value = actual
        canonical_logic.append(ast.unparse(tree))
    required_logic = [make_trade_rule(expression) for expression in canonical_logic]


    # ============================================================
    # STAGE 4 — GPT PLAYER-POOL REDUCTION AND PRIORITIZATION
    # Separate desired/required acquisitions from permitted/preferred exits.
    # No CBA evaluation is performed in this GPT stage.
    # ============================================================
    
    NHL_PROMPT_3 = 'You build FOUR DIFFERENT player pools for an NHL trade search.\n\nUSER PREFERENCES:\n{demands}\n\nPLAYER PROFILES AND THEIR CURRENT TEAMS:\n{all_results}\n\nUsing ONLY supplied team rosters, player attributes, strengths, weaknesses,\nand parsed user preferences, produce EXACTLY FOUR raw Python dictionaries,\nin the following order. Each dictionary maps exact existing NHL team names\nto lists of exact existing player names. No duplicates, fabricated names,\nor empty teams.\n\nDICTIONARY 1: required_targets\n- Players on OTHER teams who are required acquisition candidates.\n- Include every explicitly mandatory acquisition and every member of an explicit\n  mandatory OR-alternative. An OR group does NOT require each member separately;\n  the separate Required Trade Logic list decides acceptable combinations.\n- Do not classify names as required merely because the user mentions them.\n- A required named player overrides general player-type avoidance, but not\n  an explicit banned-team rule unless the user provides an exception.\n\nDICTIONARY 2: tradeable_targets\n- Players on OTHER teams who are actually desirable optional acquisitions,\n  based on nonmandatory named interests or requested position, role, skill,\n  strength, or attribute.\n- Possible desired roles include: top-six winger, bottom-six forward, two-way center, faceoff specialist, playmaking center, goal-scoring winger, power-play specialist, penalty killer, shutdown defenseman, puck-moving defenseman, offensive defenseman, physical defenseman, right-shot defenseman, depth defenseman, starting goaltender, backup goaltender, net-front forward, strong skater, reliable defensive forward.\n- Select meaningful semantic matches supported by player profiles; do not stretch\n  vague similarities or simply include every otherwise-allowable player.\n- Exclude banned teams, explicitly unwanted players, and player types the\n  user rejects, subject to explicit named-player exceptions.\n- Exclude everyone already in required_targets.\n\nDICTIONARY 3: wanted_to_trade_away\n- Players on the USER\'S OWN team whom the user affirmatively wants to move.\n- Desire to move is distinct from mere willingness to move.\n- Exclude untouchables unless explicitly overridden by the user.\n\nDICTIONARY 4: can_trade_away\n- Remaining players on the USER\'S OWN team the user is willing to move.\n- Exclude named untouchables and those matching protected player types.\n- Interpret player-type restrictions using the supplied profiles rather than\n  relying on exact keyword matching.\n- If the user protects good two-way forwards, use supplied profiles to identify players with substantial offense AND defensive awareness, defensive-zone coverage, forechecking or penalty-killing value. If the user protects strong goaltenders, use supplied goaltending strengths rather than scoring categories.\n- Include named players explicitly permitted to move, even if they overlap a\n  broad protected type, where the user\'s explicit permission overrides it.\n- Include the rest of the nonprotected user-team roster unless the user says\n  ONLY certain players may be moved.\n- Exclude anyone already in wanted_to_trade_away.\n\nIMPORTANT FILTERING AND LOGIC RULES:\n1. First identify the user\'s own team and any excluded opposing teams.\n2. Apply user-team untouchable name and player-type restrictions.\n3. Apply other-team exclusions for names, teams, and undesired player types.\n   Examples of undesired characteristics (only when supported by profiles):\n   poor defensive positioning, weak skating, inconsistent puck control, low offensive creation, frequent turnovers, weak rebound control or poor high-danger goaltending.\n4. Keep explicitly named wanted players despite GENERAL skill/attribute avoidance\n   when their team is not banned.\n5. Construct the four pools with priority: required_targets, tradeable_targets,\n   wanted_to_trade_away, can_trade_away.\n6. The FOUR pools must be PAIRWISE DISJOINT by team/player identity.\n7. Include all players explicitly referenced in mandatory acquisition alternatives\n   in required_targets (unless disallowed by an explicit team ban).\n8. Do NOT apply AND/OR/NOT as if each referenced player were independently required.\n   That job belongs to the separate Required Trade Logic list already in\n   USER PREFERENCES. Do NOT emit a fifth object or repeat the logic here.\n9. Apply only preference-based reduction. Do NOT assume that any player is\n   legally tradeable or run salary-cap, contract, waiver, consent, draft-pick,\n   or league-specific transaction validation.\n10. Do NOT use outside knowledge, perform web searches, change a player\'s team,\n    fabricate a name, or create duplicate players.\n\nOUTPUT FORMAT:\nReturn exactly FOUR raw Python dictionary expressions (not assignments),\nin this strict order:\n1. required_targets\n2. tradeable_targets\n3. wanted_to_trade_away\n4. can_trade_away\n\nExample (illustrative placeholder names only):\n{"Opposing Team": ["Required Player"]}\n{"Other Team": ["Interesting Player"]}\n{"My Team": ["Player To Move"]}\n{"My Team": ["Other Allowed Player"]}\n\nNo headings, labels, comments, explanations, code fences, introductory text,\nconcluding text, or text between the four dictionaries.'
    response = client.responses.create(
        model="gpt-5.6",
        input=(NHL_PROMPT_3.replace("{demands}", demands.output_text)
                             .replace("{all_results}", "\n\n".join(all_results))),
    )

    def parse_four_dictionaries(text):
        try:
            tree = ast.parse(text.strip(), mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"GPT returned invalid pool syntax:\n{text}") from exc

        values = []
        for node in tree.body:
            if not isinstance(node, ast.Expr):
                raise ValueError(f"Unexpected non-dictionary expression:\n{text}")
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError("Expected exactly four dictionaries")
            values.append(value)
        if len(values) != 4:
            raise ValueError(f"Expected four dictionaries, got {len(values)}:\n{text}")
        return values

    required_targets, tradeable_targets, wanted_to_trade_away, can_trade_away = (
        parse_four_dictionaries(response.output_text)
    )

    # Validate GPT's names against the scraped roster: models can misspell or
    # assign players to the wrong team. Keep only verified membership.
    roster_lookup = {
        team: {normalize_name(name).casefold(): name for name in names}
        for team, names in league_players.items()
    }

    def verify_roster_pool(pool, pool_name):
        clean = {}
        for team, names in pool.items():
            if team not in roster_lookup:
                raise ValueError(f"{pool_name}: unknown team {team!r}")
            if not isinstance(names, list):
                raise ValueError(f"{pool_name}: player values must be lists")
            verified = []
            for name in names:
                actual = roster_lookup[team].get(normalize_name(name).casefold())
                if actual is None:
                    raise ValueError(f"{pool_name}: {name!r} not found on {team}")
                verified.append(actual)
            if verified:
                clean[team] = sorted(set(verified))
        return clean

    required_targets = verify_roster_pool(required_targets, "required_targets")
    tradeable_targets = verify_roster_pool(tradeable_targets, "tradeable_targets")
    wanted_to_trade_away = verify_roster_pool(wanted_to_trade_away, "wanted_to_trade_away")
    can_trade_away = verify_roster_pool(can_trade_away, "can_trade_away")

    # Favor the higher-priority group if GPT accidentally duplicates a player.
    seen = set()
    def deduplicate_pool(pool):
        cleaned = {}
        for team, players in pool.items():
            unique = [name for name in players if (team, name) not in seen]
            seen.update((team, name) for name in unique)
            if unique:
                cleaned[team] = unique
        return cleaned

    required_targets = deduplicate_pool(required_targets)
    tradeable_targets = deduplicate_pool(tradeable_targets)
    wanted_to_trade_away = deduplicate_pool(wanted_to_trade_away)
    can_trade_away = deduplicate_pool(can_trade_away)

    print("\nREQUIRED TARGETS (BEFORE CBA):", required_targets)
    print("\nTRADEABLE TARGETS (BEFORE CBA):", tradeable_targets)
    print("\nWANTED TO TRADE AWAY (BEFORE CBA):", wanted_to_trade_away)
    print("\nCAN TRADE AWAY (BEFORE CBA):", can_trade_away)

    # Check that mandatory names can be sourced from the appropriate pools.
    # This is an early feasibility diagnostic; no NHL transaction legality is claimed.
    all_incoming = {p for pool in (required_targets, tradeable_targets) for players in pool.values() for p in players}
    all_outgoing = {p for pool in (wanted_to_trade_away, can_trade_away) for players in pool.values() for p in players}
    for expression in canonical_logic:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                name = node.left.value
                side = node.comparators[0].id
                # Negative membership may intentionally refer to a player that
                # the existing filters already prevent from moving.
                if isinstance(node.ops[0], ast.In):
                    if side == "acquired" and name not in all_incoming:
                        print(f"WARNING: logic references acquisition outside filtered target pools: {name}")
                    if side == "outgoing" and name not in all_outgoing:
                        print(f"WARNING: logic references outgoing player outside filtered pools: {name}")

    print("\nFINAL NHL PLAYER DICTIONARIES — AFTER USER-PREFERENCE FILTERING")
    print("\nREQUIRED TARGETS:", required_targets)
    print("\nTRADEABLE TARGETS:", tradeable_targets)
    print("\nWANTED TO TRADE AWAY:", wanted_to_trade_away)
    print("\nCAN TRADE AWAY:", can_trade_away)
    print("\nREQUIRED TRADE LOGIC:", [rule.expression for rule in required_logic])
    return required_targets, tradeable_targets, wanted_to_trade_away, can_trade_away, required_logic

