(define (domain learned_action_schema_fragment)
  (:requirements
    :strips
    :typing
    :probabilistic-effects
  )
  (:types
    movable_item fixed_item - object
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
  )

  ;; Action schemas learned during consolidation.
  ;; Manipulation actions learned from trajectories. close the {param_1} on the left
  (:action close_object_on_left
    :parameters (?arg0 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=[], del=['open(?arg0)']
        ; bucket: close_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (not (open ?arg0)))
      )
  )

  ;; close the {param_1} on the right
  (:action close_object_on_right
    :parameters (?arg0 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=[], del=['open(?arg0)']
        ; bucket: close_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (not (open ?arg0)))
      )
  )

  ;; look into the {param_1} in front of the {param_2} on the left
  (:action look_into_object_in_front_of_object_on_left
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition (and)
    :effect (and)
  )

  ;; look into the {param_1} in front of the {param_2} on the right
  (:action look_into_object_in_front_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition (and)
    :effect (and)
  )

  ;; look into the {param_1} on the left
  (:action look_into_object_on_left
    :parameters (?arg0 - drawer)
    :precondition (and)
    :effect (and)
  )

  ;; look into the {param_1} on the right
  (:action look_into_object_on_right
    :parameters (?arg0 - drawer)
    :precondition (and)
    :effect (and)
  )

  ;; open the {param_1} on the left
  (:action open_object_on_left
    :parameters (?arg0 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['open(?arg0)'], del=[]
        ; bucket: open_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (open ?arg0))
      )
  )

  ;; open the {param_1} on the right
  (:action open_object_on_right
    :parameters (?arg0 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['open(?arg0)'], del=[]
        ; bucket: open_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (open ?arg0))
      )
  )

  ;; pick up the {param_1} in front of the {param_2} on the left
  (:action pick_up_object_in_front_of_object_on_left
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in_front_of(?arg0,?arg1)']
        ; bucket: pick_up_object_in_front_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in_front_of ?arg0 ?arg1)))
      )
  )

  ;; pick up the {param_1} in front of the {param_2} on the right
  (:action pick_up_object_in_front_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in_front_of(?arg0,?arg1)']
        ; bucket: pick_up_object_in_front_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in_front_of ?arg0 ?arg1)))
      )
  )

  ;; pick up the {param_1} in the {param_2} on the left
  (:action pick_up_object_in_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in(?arg0,?arg1)']
        ; bucket: pick_up_object_in_object_on_left_success, success: true, variant_rank: 0
        0.636364 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in ?arg0 ?arg1)))
        ; bucket: pick_up_object_in_object_on_left_failure, success: false, variant_rank: 0
        0.363636 (and)
      )
  )

  ;; pick up the {param_1} in the {param_2} on the right
  (:action pick_up_object_in_object_on_right
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_holding(?arg0)'], del=['gripper_empty()', 'in(?arg0,?arg1)']
        ; bucket: pick_up_object_in_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_holding ?arg0) (not (gripper_empty)) (not (in ?arg0 ?arg1)))
      )
  )

  ;; place the {param_1} into the {param_2} on the left
  (:action place_object_into_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'in(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_into_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (in ?arg0 ?arg1) (not (gripper_holding ?arg0)))
      )
  )

  ;; place the {param_1} into the {param_2} on the right
  (:action place_object_into_object_on_right
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'in(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_into_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (in ?arg0 ?arg1) (not (gripper_holding ?arg0)))
      )
  )

  ;; place the {param_1} on the top of the {param_2} on the left
  (:action place_object_on_top_of_object_on_left
    :parameters (?arg0 - block ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'on_top_of(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_on_top_of_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (on_top_of ?arg0 ?arg1) (not (gripper_holding ?arg0)))
      )
  )

  ;; place the {param_1} on the top of the {param_2} on the right
  (:action place_object_on_top_of_object_on_right
    :parameters (?arg0 - cup ?arg1 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['gripper_empty()', 'on_top_of(?arg0,?arg1)'], del=['gripper_holding(?arg0)']
        ; bucket: place_object_on_top_of_object_on_right_success, success: true, variant_rank: 0
        1.000000 (and (gripper_empty) (on_top_of ?arg0 ?arg1) (not (gripper_holding ?arg0)))
      )
  )

  ;; push the {param_1} on the {param_2} on the right to the {param_3} on the left
  (:action push_object_on_object_on_right_to_object_on_left
    :parameters (?arg0 - box ?arg1 - drawer ?arg2 - drawer)
    :precondition (and)
    :effect
      (probabilistic
        ; fixed: add=['on_top_of(?arg0,?arg2)'], del=['on_top_of(?arg0,?arg1)']
        ; bucket: push_object_on_object_on_right_to_object_on_left_success, success: true, variant_rank: 0
        1.000000 (and (on_top_of ?arg0 ?arg2) (not (on_top_of ?arg0 ?arg1)))
      )
  )

)
