(0,0) - center of field

How the robot is tracked over time?
1. Two mapped markers provide the first absolute pose
2. Velocity commands predict movement from that pose.
3. New marker observations correct the prediction
4. With fewer than two markers, prediction continues while commands are recent
5. Without fresh observations or commands, the stored pose becomes old



using 2 markers' field coordinates (known):

p_robot = p1 + x_e + y_n

(e points along the marker baseline and n is perpendicular to it)


## Final project
The goal is to let a rover get to a point in the field.
The position of the rubber duck (target) will be one of five possible position,
and the point where it will be placed can be provided to the algorithm
and it shouldn't be determined by the algorithm.

The markers are used to localize the rover and triangulate the position of the rover on the field,
as well as the orientation of the rover.

The rover will start in one point of the field, and the goal is to get to the rubber duck.
It will reimplement the obstacle avoidance algorithms used for the previous assignment, but it will be adapted to the task.
The rover will start going straight, edge obstacles, and use the odometry in combination with the marker detection and triangulation
to better determine the location in the field. At any moment, when the rover is in the cruise phase, it will follow the line connecting the current position and the goal (position of the duck).

We stop exactly a few centimeters before the rubber duck.