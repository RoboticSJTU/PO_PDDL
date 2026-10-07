(define (problem episode14_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    pink_cup - cup
    green_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in_front_of pink_cup green_drawer)
    (not (gripper_holding pink_cup))
    (not (on_top_of pink_cup green_drawer))
    (not (open green_drawer))
  )
  (:goal
    (and (on_top_of pink_cup green_drawer) (open green_drawer))
  )
)
