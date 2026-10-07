(define (problem episode8_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    green_cup pink_cup - cup
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (has_dark_liquid green_cup)
    (in_front_of green_cup green_drawer)
    (in_front_of pink_cup yellow_drawer)
    (not (gripper_holding green_cup))
    (not (on_top_of green_cup green_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (on_top_of green_cup green_drawer))
  )
)
