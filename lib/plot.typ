#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot

#let coordinate-system(
  x-min: -3,
  x-max: 3,
  y-min: -3,
  y-max: 3,
  size: (6, 5),
  axis-style: "school-book",
  body,
) = {
  cetz.canvas({
    import cetz.draw: *

    plot.plot(
      size: size,

      x-min: x-min,
      x-max: x-max,
      y-min: y-min,
      y-max: y-max,

      x-tick-step: 1,
      y-tick-step: 1,
      axis-style: axis-style,

      body,
    )
  })
}

#let vector(
  from,
  to,
  color: black,
  label: none,
  label-position: "center",
) = {
  import cetz.draw: *

  plot.annotate(
    resize: false,
    {
      line(
        from,
        to,
        stroke: color,
        mark: (end: ">"),
      )

      if label != none {
        let pos = if label-position == "center" {
          (
            (from.at(0) + to.at(0)) / 2,
            ((from.at(1) + to.at(1)) / 2) + 0.2,
          )
        } else if label-position == "head" {
          to
        } else {
          panic("label-position must be \"center\" or \"head\"")
        }

        content(
          pos,
          label,
          anchor: if label-position == "head" {
            "west"
          } else {
            "south"
          },
        )
      }
    },
  )
}

#let custom-plot(
  title: none,
  x-label: none,
  y-label: none,
  x-min: 0,
  x-max: 6,
  y-min: 0,
  y-max: 6,
  x-tick-step: none,
  y-tick-step: none,
  legend: none,
  size: (6, 5),
  body,
) = {
  cetz.canvas({
    import cetz.draw: *

    plot.plot(
      size: size,
      x-min: x-min,
      x-max: x-max,
      y-min: y-min,
      y-max: y-max,
      x-label: x-label,
      y-label: y-label,
      x-tick-step: x-tick-step,
      y-tick-step: y-tick-step,
      axis-style: "scientific",
      legend: legend,
      body,
    )
  })
}
