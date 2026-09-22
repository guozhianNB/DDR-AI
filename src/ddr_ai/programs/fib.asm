; ===============================================================
; fib.asm —— 滚动变量算前 8 个斐波那契数，存进内存。
;
; 只有 4 个寄存器，同时要握住 a、b、指针、计数四件事，
; 所以每一步都得往内存里跑一趟。这正是本项目想展示的效果。
;
; 内存布局：
;   8 = a       9 = b       10 = 计数
;   16.. = 结果数组
;
; 每一步的内存访问：
;   LOAD b 两次、LOAD a 两次、STORE a、STORE b、STORE 结果、
;   LOAD 计数、STORE 计数 —— 大约 10 趟。
; 8 轮下来近 80 次访问，每一次 AI 都要答一遍。
; ===============================================================

        MOVI r0, 1
        STORE 8, r0          ; a = 1
        STORE 9, r0          ; b = 1
        MOVI r0, 8
        STORE 10, r0         ; 还剩 8 项

        MOVI r0, 16          ; r0 = 结果指针

loop:   MOVI r3, 0
        LOAD r3, 10          ; r3 = 计数
        JZ   r3, done        ; 归零就收工

        MOVI r2, 0
        LOAD r2, 8           ; r2 = a
        STI  r0, r2          ; 结果数组 ← a
        ADDI r0, 1

        MOVI r1, 0
        LOAD r1, 9           ; r1 = b
        MOVI r2, 0
        LOAD r2, 8           ; r2 = a
        ADDI r2, 0           ; 确保 r2 就是 a
        ADD  r2, r1          ; r2 = a + b → 新的 b
        MOVI r1, 0
        LOAD r1, 9           ; r1 = 旧的 b
        STORE 8, r1          ; a ← 旧 b
        STORE 9, r2          ; b ← a + b

        MOVI r3, 0
        LOAD r3, 10          ; 取出计数
        SUBI r3, 1
        STORE 10, r3         ; 计数减一
        JMP  loop

done:   HALT
