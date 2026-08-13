(define (domain learned_manipulation_fragment)
          (:requirements
    :strips
    :typing
    :probabilistic-effects
    :negative-preconditions
    :universal-preconditions
    :fluents
  )

  (:functions
    (total-reward)                    ; cumulative reward
  )

    (:types
    movable_item fixed_item last_action_marker - object
    block box cup - movable_item
    drawer - fixed_item
  )

  (:constants
    close_object_on_left_success_0 - last_action_marker
    close_object_on_right_success_0 - last_action_marker
    look_into_object_in_front_of_object_on_left_success_0 - last_action_marker
    look_into_object_in_front_of_object_on_right_success_0 - last_action_marker
    look_into_object_on_left_success_0 - last_action_marker
    look_into_object_on_right_success_0 - last_action_marker
    open_object_on_left_success_0 - last_action_marker
    open_object_on_right_success_0 - last_action_marker
    pick_up_object_in_front_of_object_on_left_success_0 - last_action_marker
    pick_up_object_in_front_of_object_on_right_success_0 - last_action_marker
    pick_up_object_in_object_on_left_fail_0 - last_action_marker
    pick_up_object_in_object_on_left_success_0 - last_action_marker
    pick_up_object_in_object_on_right_success_0 - last_action_marker
    place_object_into_object_on_left_success_0 - last_action_marker
    place_object_into_object_on_right_success_0 - last_action_marker
    place_object_on_top_of_object_on_left_success_0 - last_action_marker
    place_object_on_top_of_object_on_right_success_0 - last_action_marker
    push_object_on_object_on_right_to_object_on_left_success_0 - last_action_marker
  )

      (:predicates
    ;; The gripper is not holding any object.
    (gripper_empty)
    ;; The gripper is currently holding the object.
    (gripper_holding ?x0 - movable_item)
    ;; The cup contains dark liquid.
    (has_dark_liquid ?x0 - cup)
    ;; The first object is inside the second object.
    (in ?x0 - movable_item ?x1 - drawer)
    ;; The first object is positioned in front of the second object.
    (in_front_of ?x0 - movable_item ?x1 - fixed_item)
    ;; The first object rests on top of the second object.
    (on_top_of ?x0 - movable_item ?x1 - fixed_item)
    ;; The drawer is open.
    (open ?x0 - drawer)
    (last_action_1_param ?x0 - last_action_marker ?x1 - object)
    (last_action_2_param ?x0 - last_action_marker ?x1 - object ?x2 - object)
    (last_action_3_param ?x0 - last_action_marker ?x1 - object ?x2 - object ?x3 - object)
  )

  (:observables
    (obs-nothing)
    (obs_in ?arg0 - movable_item ?arg1 - drawer)
    (obs_has_dark_liquid ?arg0 - cup)
  )

  (:action close_object_on_left
    :parameters (?arg0 - drawer)
    :precondition
      (and (gripper_empty) (open ?arg0))
    :effect
      (and
        (decrease (total-reward) 22.9)
        (probabilistic
        ; fixed: add=[], del=['open(?arg0)']
        ; bucket: close_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (not (open ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param close_object_on_left_success_0 ?arg0))
      )
    )
  )

  (:action close_object_on_right
    :parameters (?arg0 - drawer)
    :precondition
      (and (gripper_empty) (open ?arg0))
    :effect
      (and
        (decrease (total-reward) 22.9)
        (probabilistic
        ; fixed: add=[], del=['open(?arg0)']
        ; bucket: close_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (not (open ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param close_object_on_right_success_0 ?arg0))
      )
    )
  )

  (:action open_object_on_left
    :parameters (?arg0 - drawer)
    :precondition
      (and (gripper_empty) (not (open ?arg0)) (forall (?dep0_0 - movable_item) (and (not (in_front_of ?dep0_0 ?arg0)))))
    :effect
      (and
        (decrease (total-reward) 25.645455)
        (probabilistic
        ; fixed: add=['open(?arg0)'], del=[]
        ; bucket: open_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (open ?arg0) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param open_object_on_left_success_0 ?arg0))
      )
    )
  )

  (:action open_object_on_right
    :parameters (?arg0 - drawer)
    :precondition
      (and (gripper_empty) (not (open ?arg0)) (forall (?dep0_0 - movable_item) (and (not (in_front_of ?dep0_0 ?arg0)))))
    :effect
      (and
        (decrease (total-reward) 24.52)
        (probabilistic
        ; fixed: add=['open(?arg0)'], del=[]
        ; bucket: open_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (open ?arg0) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param open_object_on_right_success_0 ?arg0))
      )
    )
  )

  (:action pick_up_object_in_front_of_object_on_left
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and (gripper_empty) (in_front_of ?arg0 ?arg1))
    :effect
      (and
        (decrease (total-reward) 22.5625)
        (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in_front_of(?arg0,?arg1)']
        ; bucket: pick_up_object_in_front_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in_front_of ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_front_of_object_on_left_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action pick_up_object_in_front_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and (gripper_empty) (in_front_of ?arg0 ?arg1) (forall (?dep0_0 - movable_item) (and (not (on_top_of ?dep0_0 ?arg1)))))
    :effect
      (and
        (decrease (total-reward) 22.35)
        (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in_front_of(?arg0,?arg1)']
        ; bucket: pick_up_object_in_front_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in_front_of ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_front_of_object_on_right_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action pick_up_object_in_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and (gripper_empty) (in ?arg0 ?arg1) (open ?arg1))
    :effect
      (and
        (decrease (total-reward) 23.536364)
        (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in(?arg0,?arg1)']
        ; bucket: pick_up_object_in_object_on_left_success, success: true, variant_rank: 0
        0.636364 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_object_on_left_success_0 ?arg0 ?arg1))
        ; bucket: pick_up_object_in_object_on_left_failure, success: false, variant_rank: 0
        0.363636 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_object_on_left_fail_0 ?arg0 ?arg1))
      )
    )
  )

  (:action pick_up_object_in_object_on_right
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and (gripper_empty) (in ?arg0 ?arg1) (open ?arg1))
    :effect
      (and
        (decrease (total-reward) 23.475)
        (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in(?arg0,?arg1)']
        ; bucket: pick_up_object_in_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_object_on_right_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action place_object_into_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and (gripper_holding ?arg0) (open ?arg1))
    :effect
      (and
        (decrease (total-reward) 20.85)
        (probabilistic
        ; fixed: add=['gripper_empty()', 'in(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_into_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (in ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_into_object_on_left_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action place_object_into_object_on_right
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and (gripper_holding ?arg0) (open ?arg1))
    :effect
      (and
        (decrease (total-reward) 22.666667)
        (probabilistic
        ; fixed: add=['gripper_empty()', 'in(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_into_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (in ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_into_object_on_right_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action place_object_on_top_of_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and (gripper_holding ?arg0))
    :effect
      (and
        (decrease (total-reward) 20.65)
        (probabilistic
        ; fixed: add=['gripper_empty()', 'on_top_of(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_on_top_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (on_top_of ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_on_top_of_object_on_left_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action place_object_on_top_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and (gripper_holding ?arg0) (not (open ?arg1)) (forall (?dep0_0 - movable_item) (and (not (on_top_of ?dep0_0 ?arg1)))))
    :effect
      (and
        (decrease (total-reward) 24.13125)
        (probabilistic
        ; fixed: add=['gripper_empty()', 'on_top_of(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_on_top_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (on_top_of ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_on_top_of_object_on_right_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action push_object_on_object_on_right_to_object_on_left
    :parameters (?arg0 - box ?arg1 - drawer ?arg2 - drawer)
    :precondition
      (and (gripper_empty) (not (open ?arg1)) (not (open ?arg2)) (on_top_of ?arg0 ?arg1))
    :effect
      (and
        (decrease (total-reward) 21.925)
        (probabilistic
        ; fixed: add=['on_top_of(?arg0,?arg2)'], del=['on_top_of(?arg0,?arg1)']
        ; bucket: push_object_on_object_on_right_to_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (on_top_of ?arg0 ?arg2) (not (on_top_of ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_3_param push_object_on_object_on_right_to_object_on_left_success_0 ?arg0 ?arg1 ?arg2))
      )
    )
  )

  (:action look_into_object_in_front_of_object_on_left
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and (gripper_empty) (in_front_of ?arg0 ?arg1))
    :effect
      (and
        (decrease (total-reward) 15)
        (probabilistic
        ; bucket: look_into_object_in_front_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param look_into_object_in_front_of_object_on_left_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action look_into_object_in_front_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and (gripper_empty) (in_front_of ?arg0 ?arg1))
    :effect
      (and
        (decrease (total-reward) 15)
        (probabilistic
        ; bucket: look_into_object_in_front_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param look_into_object_in_front_of_object_on_right_success_0 ?arg0 ?arg1))
      )
    )
  )

  (:action look_into_object_on_left
    :parameters (?arg0 - drawer)
    :precondition
      (and (gripper_empty) (open ?arg0))
    :effect
      (and
        (decrease (total-reward) 15)
        (probabilistic
        ; bucket: look_into_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param look_into_object_on_left_success_0 ?arg0))
      )
    )
  )

  (:action look_into_object_on_right
    :parameters (?arg0 - drawer)
    :precondition
      (and (gripper_empty) (open ?arg0))
    :effect
      (and
        (decrease (total-reward) 15)
        (probabilistic
        ; bucket: look_into_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param look_into_object_on_right_success_0 ?arg0))
      )
    )
  )

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

(:observation init_obs_has_dark_liquid_true
  :parameters (?obs0 - cup)
  :condition (and (has_dark_liquid ?obs0) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))))
  :distribution (probabilistic
      0.857143 (obs_has_dark_liquid ?obs0)
      0.142857 (not (obs_has_dark_liquid ?obs0))
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
)
