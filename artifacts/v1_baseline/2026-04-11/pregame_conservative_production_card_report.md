# Pregame Conservative Production Card - 2026-04-11

- Raw candidate rows: 62
- After strict gate + vendor depth: 44
- After conservative main-line filter: 32
- After exact cross-book confirmation: 32
- Deduped conservative pool: 5
- Final conservative card: 5

Rules: mainline_distance <= 0.5, exact_book_count >= 3, one best vendor per exact bet, one best bet per pitcher

## Final conservative production card

|   game_id |   pitcher_id | pitcher_name        | vendor      |   line_value |   market_mean |   mainline_distance | best_side   |   best_ev |   decision_score |   conservative_score |   gate_conf_dist |   vendor_count |   top_quote_count |   exact_book_count |   model_over_prob |   market_over_prob |   model_under_prob |   market_under_prob |   selected_market_prob |   over_odds |   under_odds | passes_strict_ev_gate   | action_tier   |
|----------:|-------------:|:--------------------|:------------|-------------:|--------------:|--------------------:|:------------|----------:|-----------------:|---------------------:|-----------------:|---------------:|------------------:|-------------------:|------------------:|-------------------:|-------------------:|--------------------:|-----------------------:|------------:|-------------:|:------------------------|:--------------|
|   5057987 |           51 | Jack Leiter         | betmgm      |          5.5 |           5.5 |                   0 | under       |  0.777069 |         0.889701 |              1.9397  |         0.450526 |              7 |                10 |                  7 |         0.0494745 |           0.5      |           0.950526 |            0.5      |               0.5      |        -115 |         -115 | True                    | BET           |
|   5057986 |         1516 | German Marquez      | betonlineag |          4.5 |           4.5 |                   0 | over        |  0.530623 |         0.573455 |              1.62345 |         0.171326 |              7 |                10 |                  7 |         0.671326  |           0.41219  |           0.328674 |            0.58781  |               0.41219  |         128 |         -167 | True                    | BET           |
|   5057986 |          979 | Ryan Feltner        | draftkings  |          3.5 |           3.5 |                   0 | over        |  0.341493 |         0.385839 |              1.58584 |         0.177387 |              8 |                10 |                  8 |         0.677387  |           0.476141 |           0.322613 |            0.523859 |               0.476141 |        -102 |         -125 | True                    | BET           |
|   5057988 |         1545 | Lance McCullers Jr. | betonlineag |          5.5 |           5.5 |                   0 | over        |  0.382777 |         0.416353 |              1.36635 |         0.134302 |              7 |                10 |                  6 |         0.634302  |           0.430713 |           0.365698 |            0.569287 |               0.430713 |         118 |         -154 | True                    | BET           |
|   5057988 |          369 | Luis Castillo       | fanduel     |          5.5 |           5.5 |                   0 | over        |  0.238644 |         0.27151  |              1.07151 |         0.131466 |              8 |                11 |                  4 |         0.631466  |           0.475915 |           0.368534 |            0.524085 |               0.475915 |        -104 |         -128 | True                    | BET           |