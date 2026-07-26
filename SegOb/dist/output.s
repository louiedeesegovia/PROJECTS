.data
          x: .byte 2
          y: .byte 10
          sum: .byte 12
.code
        daddiu r1, r0, 2
        sb r1, x(r0)
        daddiu r2, r0, 10
        sb r2, y(r0)
        lb r3, x(r0)
        lb r4, y(r0)
        daddu r5, r3, r4
        sb r5, sum(r0)
