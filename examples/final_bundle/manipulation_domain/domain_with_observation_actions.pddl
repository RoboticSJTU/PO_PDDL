(define (domain learned_manipulation_fragment)
        (:requirements
    :strips
    :typing
    :probabilistic-effects
    :negative-preconditions
    :universal-preconditions
  )
    (:types
    movable_item fixed_item last_action_marker - object
    containable_item - fixed_item
    block box cup - movable_item
    drawer - containable_item
  )

          (:predicates
    ;; The gripper is not holding any object.
    (gripper_empty)
    ;; The gripper is currently holding the object.
    (gripper_holding ?x0 - movable_item)
    ;; The cup contains dark liquid.
    (has_dark_liquid ?x0 - cup)
    ;; The first object is inside the second object.
    (in ?x0 - movable_item ?x1 - containable_item)
    ;; The first object is positioned in front of the second object.
    (in_front_of ?x0 - movable_item ?x1 - fixed_item)
    ;; The drawer occupies the left-side position.
    (on_left ?x0 - drawer)
    ;; The first object rests on top of the second object.
    (on_top_of ?x0 - movable_item ?x1 - fixed_item)
    ;; The drawer is open.
    (open ?x0 - drawer)
    (last_action_1_param ?x0 - last_action_marker ?x1 - object)
    (last_action_2_param ?x0 - last_action_marker ?x1 - object ?x2 - object)
    (last_action_3_param ?x0 - last_action_marker ?x1 - object ?x2 - object ?x3 - object)
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


  ;; Manipulation actions learned from trajectories. Manipulation actions learned from trajectories.
  ;; close the {param_1} on the left
  (:action close_object_on_left
    :parameters (?arg0 - drawer)
    :precondition
      (and
        (gripper_empty)
        (on_left ?arg0)
        (open ?arg0)
      )
    :effect
      (probabilistic
        ; fixed: add=[], del=['open(?arg0)']
        ; bucket: close_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (not (open ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param close_object_on_left_success_0 ?arg0))
      )
  )

  ;; close the {param_1} on the right
  (:action close_object_on_right
    :parameters (?arg0 - drawer)
    :precondition
      (and
        (gripper_empty)
        (not (on_left ?arg0))
        (open ?arg0)
      )
    :effect
      (probabilistic
        ; fixed: add=[], del=['open(?arg0)']
        ; bucket: close_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (not (open ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param close_object_on_right_success_0 ?arg0))
      )
  )

  ;; open the {param_1} on the left
  (:action open_object_on_left
    :parameters (?arg0 - drawer)
    :precondition
      (and
        (gripper_empty)
        (not (open ?arg0))
        (on_left ?arg0)
        (forall (?dep0_0 - movable_item)
          (and
            (not (in_front_of ?dep0_0 ?arg0))
          )
        )
      )
    :effect
      (probabilistic
        ; fixed: add=['open(?arg0)'], del=[]
        ; bucket: open_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (open ?arg0) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param open_object_on_left_success_0 ?arg0))
      )
  )

  ;; open the {param_1} on the right
  (:action open_object_on_right
    :parameters (?arg0 - drawer)
    :precondition
      (and
        (gripper_empty)
        (not (on_left ?arg0))
        (not (open ?arg0))
        (forall (?dep0_0 - movable_item)
          (and
            (not (in_front_of ?dep0_0 ?arg0))
          )
        )
      )
    :effect
      (probabilistic
        ; fixed: add=['open(?arg0)'], del=[]
        ; bucket: open_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (open ?arg0) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param open_object_on_right_success_0 ?arg0))
      )
  )

  ;; pick up the {param_1} in front of the {param_2} on the left
  (:action pick_up_object_in_front_of_object_on_left
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and
        (gripper_empty)
        (in_front_of ?arg0 ?arg1)
        (on_left ?arg1)
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in_front_of(?arg0,?arg1)']
        ; bucket: pick_up_object_in_front_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in_front_of ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_front_of_object_on_left_success_0 ?arg0 ?arg1))
      )
  )

  ;; pick up the {param_1} in front of the {param_2} on the right
  (:action pick_up_object_in_front_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and
        (gripper_empty)
        (in_front_of ?arg0 ?arg1)
        (not (on_left ?arg1))
        (forall (?dep0_0 - movable_item)
          (and
            (not (on_top_of ?dep0_0 ?arg1))
          )
        )
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in_front_of(?arg0,?arg1)']
        ; bucket: pick_up_object_in_front_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in_front_of ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_front_of_object_on_right_success_0 ?arg0 ?arg1))
      )
  )

  ;; pick up the {param_1} in the {param_2} on the left
  (:action pick_up_object_in_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and
        (gripper_empty)
        (in ?arg0 ?arg1)
        (on_left ?arg1)
        (open ?arg1)
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in(?arg0,?arg1)']
        ; bucket: pick_up_object_in_object_on_left_success, success: true, variant_rank: 0
        0.636364 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_object_on_left_success_0 ?arg0 ?arg1))
        ; bucket: pick_up_object_in_object_on_left_failure, success: false, variant_rank: 0
        0.363636 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_object_on_left_fail_0 ?arg0 ?arg1))
      )
  )

  ;; pick up the {param_1} in the {param_2} on the right
  (:action pick_up_object_in_object_on_right
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and
        (gripper_empty)
        (in ?arg0 ?arg1)
        (not (on_left ?arg1))
        (open ?arg1)
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in(?arg0,?arg1)']
        ; bucket: pick_up_object_in_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param pick_up_object_in_object_on_right_success_0 ?arg0 ?arg1))
      )
  )

  ;; place the {param_1} into the {param_2} on the left
  (:action place_object_into_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and
        (gripper_holding ?arg0)
        (on_left ?arg1)
        (open ?arg1)
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'in(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_into_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (in ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_into_object_on_left_success_0 ?arg0 ?arg1))
      )
  )

  ;; place the {param_1} into the {param_2} on the right
  (:action place_object_into_object_on_right
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and
        (gripper_holding ?arg0)
        (not (on_left ?arg1))
        (open ?arg1)
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'in(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_into_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (in ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_into_object_on_right_success_0 ?arg0 ?arg1))
      )
  )

  ;; place the {param_1} on the top of the {param_2} on the left
  (:action place_object_on_top_of_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition
      (and
        (gripper_holding ?arg0)
        (on_left ?arg1)
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'on_top_of(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_on_top_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (on_top_of ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_on_top_of_object_on_left_success_0 ?arg0 ?arg1))
      )
  )

  ;; place the {param_1} on the top of the {param_2} on the right
  (:action place_object_on_top_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and
        (gripper_holding ?arg0)
        (not (on_left ?arg1))
        (not (open ?arg1))
        (forall (?dep0_0 - movable_item)
          (and
            (not (on_top_of ?dep0_0 ?arg1))
          )
        )
      )
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'on_top_of(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_on_top_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (on_top_of ?arg0 ?arg1) (not (gripper_holding ?arg0)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param place_object_on_top_of_object_on_right_success_0 ?arg0 ?arg1))
      )
  )

  ;; push the {param_1} on the {param_2} on the right to the {param_3} on the left
  (:action push_object_on_object_on_right_to_object_on_left
    :parameters (?arg0 - box ?arg1 - drawer ?arg2 - drawer)
    :precondition
      (and
        (gripper_empty)
        (not (on_left ?arg1))
        (not (open ?arg1))
        (not (open ?arg2))
        (on_left ?arg2)
        (on_top_of ?arg0 ?arg1)
      )
    :effect
      (probabilistic
        ; fixed: add=['on_top_of(?arg0,?arg2)'], del=['on_top_of(?arg0,?arg1)']
        ; bucket: push_object_on_object_on_right_to_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (on_top_of ?arg0 ?arg2) (not (on_top_of ?arg0 ?arg1)) (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_3_param push_object_on_object_on_right_to_object_on_left_success_0 ?arg0 ?arg1 ?arg2))
      )
  )

  ;; Observation-action learning is not implemented yet.

  (:action look_into_object_in_front_of_object_on_left
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and
        (gripper_empty)
        (in_front_of ?arg0 ?arg1)
        (on_left ?arg1)
      )
    :effect
      (probabilistic
        ; bucket: look_into_object_in_front_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param look_into_object_in_front_of_object_on_left_success_0 ?arg0 ?arg1))
      )
  )

  (:action look_into_object_in_front_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition
      (and
        (gripper_empty)
        (in_front_of ?arg0 ?arg1)
        (not (on_left ?arg1))
      )
    :effect
      (probabilistic
        ; bucket: look_into_object_in_front_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_2_param look_into_object_in_front_of_object_on_right_success_0 ?arg0 ?arg1))
      )
  )

  (:action look_into_object_on_left
    :parameters (?arg0 - drawer)
    :precondition
      (and
        (gripper_empty)
        (on_left ?arg0)
        (open ?arg0)
      )
    :effect
      (probabilistic
        ; bucket: look_into_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param look_into_object_on_left_success_0 ?arg0))
      )
  )

  (:action look_into_object_on_right
    :parameters (?arg0 - drawer)
    :precondition
      (and
        (gripper_empty)
        (not (on_left ?arg0))
        (open ?arg0)
      )
    :effect
      (probabilistic
        ; bucket: look_into_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (forall (?m - last_action_marker ?x0 - object) (not (last_action_1_param ?m ?x0))) (forall (?m - last_action_marker ?x0 - object ?x1 - object) (not (last_action_2_param ?m ?x0 ?x1))) (forall (?m - last_action_marker ?x0 - object ?x1 - object ?x2 - object) (not (last_action_3_param ?m ?x0 ?x1 ?x2))) (last_action_1_param look_into_object_on_right_success_0 ?arg0))
      )
  )
)
