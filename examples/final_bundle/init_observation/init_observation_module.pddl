;; Init observation domain module learned from initial scene descriptions
(:observables
  (obs-nothing)
  (obs_has_dark_liquid ?arg0 - cup)
)

;; init observation / predicate=has_dark_liquid
(:observation init_obs_has_dark_liquid_true
  :parameters (?obs0 - cup)
  :condition (and (has_dark_liquid ?obs0) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))))
  :distribution (probabilistic
      0.142857 (obs_has_dark_liquid ?obs0)
      0.857143 (not (obs_has_dark_liquid ?obs0))
    )
)
(:observation init_obs_has_dark_liquid_false
  :parameters (?obs0 - cup)
  :condition (and (not (has_dark_liquid ?obs0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))))
  :distribution (probabilistic
      0.000000 (obs_has_dark_liquid ?obs0)
      1.000000 (not (obs_has_dark_liquid ?obs0))
    )
)
