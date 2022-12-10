# Random-lines

There are several ways to represent a line in the plane. For stochastic geometry, it is convenient to
parameterize a line ℓ by a pair (d,θ), where d is the distance from the origin to the line and θ is the
angle of the line with the x-axis. See for example http://www.siue.edu/~aweyhau/project.html.
Consider a convex object C in the plane and another convex object C′
inside C. We want to analyze
the probability that a random line intersecting C also intersects C′. The lines should be chosen
uniformly at random with respect to the parameterization described above. The expected answer is
the ratio between the perimeters of the objects.
You may want to try different ways to select a “random line” that intersects C to see different
behaviors. Note that a good definition of random line should be invariant under translations and
rotations: for each translation or rotation ρ, it should be that the probability that a random line
intersecting ρ(C) intersects ρ(C′) is the same as the probability that a random line intersecting C
intersects C′
. In other words, the measure of lines intersecting C and ρ(C) should be the same for
any translation or rotation ρ
