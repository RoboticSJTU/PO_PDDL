(define (problem episode6_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    green_cup - cup
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in_front_of green_cup yellow_drawer)
    (not (gripper_holding green_cup))
    (not (on_top_of green_cup green_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (on_top_of green_cup green_drawer))
  )
)
