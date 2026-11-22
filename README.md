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
