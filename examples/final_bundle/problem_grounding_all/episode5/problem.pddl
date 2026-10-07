(define (problem episode5_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    green_cup pink_cup - cup
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (has_dark_liquid pink_cup)
    (in_front_of green_cup yellow_drawer)
    (in_front_of pink_cup green_drawer)
    (not (gripper_holding pink_cup))
    (not (on_top_of pink_cup green_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (on_top_of pink_cup green_drawer))
  )
)
