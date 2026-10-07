(define (problem episode15_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    green_cup - cup
    green_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in_front_of green_cup green_drawer)
    (not (gripper_holding green_cup))
    (not (on_top_of green_cup green_drawer))
    (not (open green_drawer))
  )
  (:goal
    (and (on_top_of green_cup green_drawer) (open green_drawer))
  )
)
