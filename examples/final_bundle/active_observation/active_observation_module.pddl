;; Active observation domain module learned from active perception actions
(:observables
  (obs-nothing)
  (obs_has_dark_liquid ?arg0 - cup)
  (obs_in ?arg0 - movable_item ?arg1 - containable_item)
)

;; look_into_object_in_front_of_object_on_left / look_into_object_in_front_of_object_on_left_success / predicate=has_dark_liquid
(:observation active_look_into_object_in_front_of_object_on_left_success_0_has_dark_liquid_true
  :parameters (?arg0 - cup ?arg1 - drawer)
  :condition (and ( last_action_2_param look_into_object_in_front_of_object_on_left_success_0 ?arg0 ?arg1) (has_dark_liquid ?arg0))
  :distribution (probabilistic
      1.000000 (obs_has_dark_liquid ?arg0)
      0.000000 (not (obs_has_dark_liquid ?arg0))
    )
)
(:observation active_look_into_object_in_front_of_object_on_left_success_0_has_dark_liquid_false
  :parameters (?arg0 - cup ?arg1 - drawer)
  :condition (and ( last_action_2_param look_into_object_in_front_of_object_on_left_success_0 ?arg0 ?arg1) (not (has_dark_liquid ?arg0)))
  :distribution (probabilistic
      0.000000 (obs_has_dark_liquid ?arg0)
      1.000000 (not (obs_has_dark_liquid ?arg0))
    )
)

;; look_into_object_in_front_of_object_on_right / look_into_object_in_front_of_object_on_right_success / predicate=has_dark_liquid
(:observation active_look_into_object_in_front_of_object_on_right_success_0_has_dark_liquid_true
  :parameters (?arg0 - cup ?arg1 - drawer)
  :condition (and ( last_action_2_param look_into_object_in_front_of_object_on_right_success_0 ?arg0 ?arg1) (has_dark_liquid ?arg0))
  :distribution (probabilistic
      1.000000 (obs_has_dark_liquid ?arg0)
      0.000000 (not (obs_has_dark_liquid ?arg0))
    )
)
(:observation active_look_into_object_in_front_of_object_on_right_success_0_has_dark_liquid_false
  :parameters (?arg0 - cup ?arg1 - drawer)
  :condition (and ( last_action_2_param look_into_object_in_front_of_object_on_right_success_0 ?arg0 ?arg1) (not (has_dark_liquid ?arg0)))
  :distribution (probabilistic
      0.000000 (obs_has_dark_liquid ?arg0)
      1.000000 (not (obs_has_dark_liquid ?arg0))
    )
)

;; look_into_object_on_left / look_into_object_on_left_success / predicate=in
(:observation active_look_into_object_on_left_success_0_in_true
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param look_into_object_on_left_success_0 ?arg0) (in ?obs0 ?arg0))
  :distribution (probabilistic
      1.000000 (obs_in ?obs0 ?arg0)
      0.000000 (not (obs_in ?obs0 ?arg0))
    )
)
(:observation active_look_into_object_on_left_success_0_in_false
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param look_into_object_on_left_success_0 ?arg0) (not (in ?obs0 ?arg0)))
  :distribution (probabilistic
      0.000000 (obs_in ?obs0 ?arg0)
      1.000000 (not (obs_in ?obs0 ?arg0))
    )
)

;; look_into_object_on_right / look_into_object_on_right_success / predicate=in
(:observation active_look_into_object_on_right_success_0_in_true
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param look_into_object_on_right_success_0 ?arg0) (in ?obs0 ?arg0))
  :distribution (probabilistic
      1.000000 (obs_in ?obs0 ?arg0)
      0.000000 (not (obs_in ?obs0 ?arg0))
    )
)
(:observation active_look_into_object_on_right_success_0_in_false
  :parameters (?arg0 - drawer ?obs0 - block)
  :condition (and ( last_action_1_param look_into_object_on_right_success_0 ?arg0) (not (in ?obs0 ?arg0)))
  :distribution (probabilistic
      0.000000 (obs_in ?obs0 ?arg0)
      1.000000 (not (obs_in ?obs0 ?arg0))
    )
)
