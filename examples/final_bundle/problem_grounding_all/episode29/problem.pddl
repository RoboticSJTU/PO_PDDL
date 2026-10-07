(define (problem episode29_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    green_drawer - drawer
  )
  (:init
    (gripper_empty)
    (open green_drawer)
  )
  (:goal
    (and (not (open green_drawer)))
  )
)
