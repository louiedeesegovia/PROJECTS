.data
          a: .byte 10
          b: .byte 5
          sum: .byte 0
          sum1: .byte 0
          sample: .byte 0
          m: .byte 10
          n: .byte 10
          o: .byte 5
          p: .byte 2
.code
        daddiu r1, r0, 10
        sb r1, a(r0)
        daddiu r2, r0, 5
        sb r2, b(r0)
        daddiu r3, r0, 0
        sb r3, sum(r0)
        daddiu r4, r0, 0
        sb r4, sum1(r0)
        daddiu r5, r0, 0
        sb r5, sample(r0)
        daddiu r6, r0, 10
        sb r6, m(r0)
        daddiu r7, r0, 10
        sb r7, n(r0)
        daddiu r8, r0, 5
        sb r8, o(r0)
        daddiu r9, r0, 2
        sb r9, p(r0)
        lb r10, o(r0)
        lb r11, m(r0)
        lb r12, n(r0)
        dmult r11, r12
        mflo r13
        lb r14, p(r0)
        ddiv r13, r14
        mflo r15
        daddu r16, r10, r15
        sb r16, sum1(r0)
        lb r17, m(r0)
        daddiu r18, r0, 3
        ddiv r17, r18
        mfhi r19
        sb r19, sum(r0)
