;; Passive observation domain module learned from manipulation effect variants
(:observables
  (obs-nothing)
  (obs_in ?arg0 - movable_item ?arg1 - containable_item)
)

;; open_object_on_left / open_object_on_left_success / predicate=in
(:observation passive_open_object_on_left_success_0_in_true
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param open_object_on_left_success_0 ?arg0) (in ?obs0 ?arg0))
  :distribution (probabilistic
      1.000000 (obs_in ?obs0 ?arg0)
      0.000000 (not (obs_in ?obs0 ?arg0))
    )
)
(:observation passive_open_object_on_left_success_0_in_false
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param open_object_on_left_success_0 ?arg0) (not (in ?obs0 ?arg0)))
  :distribution (probabilistic
      0.000000 (obs_in ?obs0 ?arg0)
      1.000000 (not (obs_in ?obs0 ?arg0))
    )
)

;; open_object_on_right / open_object_on_right_success / predicate=in
(:observation passive_open_object_on_right_success_0_in_true
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param open_object_on_right_success_0 ?arg0) (in ?obs0 ?arg0))
  :distribution (probabilistic
      1.000000 (obs_in ?obs0 ?arg0)
      0.000000 (not (obs_in ?obs0 ?arg0))
    )
)
(:observation passive_open_object_on_right_success_0_in_false
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param open_object_on_right_success_0 ?arg0) (not (in ?obs0 ?arg0)))
  :distribution (probabilistic
      0.000000 (obs_in ?obs0 ?arg0)
      1.000000 (not (obs_in ?obs0 ?arg0))
    )
)
