///////////////////////////////////////////////////////////////////
// Gmsh file for creating a finite element mesh
// In this case, we consider a sphere of radius R in a channel
// For a great tutorial on using Gmsh, see
// https://www.youtube.com/watch?v=aFc6Wpm69xo
///////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

// element size
es = 3e-2;

// length and half-width of the channel
L = 10;
H = 0.5;

// radius of circle
R = 0.3;

////////////////////////////////////////////////////////////

// Create all of the points

// Points for the circle
Point(1) = {-R, 0, 0, es};
Point(2) = {0, 0, 0, es};
Point(3) = {R, 0, 0, es};

// Points for the domain corners
Point(4) = {L/2, 0, 0, es};
Point(5) = {L/2, H, 0, es};
Point(6) = {-L/2, H, 0, es};
Point(7) = {-L/2, 0, 0, es};

// Create circle and lines
Circle(1) = {3, 2, 1};

Line(2) = {1, 7};
Line(3) = {7, 6};
Line(4) = {6, 5};
Line(5) = {5, 4};
Line(6) = {4, 3};

Curve Loop(1) = {1:6};
Plane Surface(1) = {1};

// create physical lines (for FEnICS)

// circle
Physical Curve(1) = {1};

// axis
Physical Curve(2) = {2, 6};

// inlet
Physical Curve(3) = {3};

// output
Physical Curve(4) = {5};

// channel wall
Physical Curve(5) = {4};

// bulk
Physical Surface(6) = {1};
