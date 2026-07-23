`timescale 1ns / 1ps
// =============================================================
// TB_ARGMAX : Kiem tra doc lap module argmax (47 lop)
// Chay nhieu vector logit ngau nhien, doi chieu voi phep tinh
// argmax tham chieu (uu tien chi so nho hon khi bang nhau).
// =============================================================
module tb_argmax;

    reg clk, rst_n, start;
    wire ready, done;

    reg         in_wr_en;
    reg [5:0]   in_wr_addr;
    reg signed [31:0] in_wr_data;

    wire [5:0] predicted_class;
    wire       valid_out;

    argmax dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .ready(ready), .done(done),
        .in_wr_en(in_wr_en), .in_wr_addr(in_wr_addr), .in_wr_data(in_wr_data),
        .predicted_class(predicted_class), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Xuat waveform ra file .vcd de xem bang GTKWave
    initial begin
        $dumpfile("argmax.vcd");
        $dumpvars(0, tb_argmax);
    end

    reg signed [31:0] logits [0:46];
    integer t, i, errors;
    integer ref_idx;
    reg signed [31:0] ref_max;

    initial begin
        rst_n = 0; start = 0; in_wr_en = 0; in_wr_addr = 0; in_wr_data = 0;
        errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;

        for (t = 0; t < 20; t = t + 1) begin
            // Sinh 47 logit ngau nhien (co the am/duong)
            for (i = 0; i < 47; i = i + 1)
                logits[i] = $random % 1000;

            // Tinh argmax tham chieu: giu chi so dau tien dat max
            ref_max = logits[0];
            ref_idx = 0;
            for (i = 1; i < 47; i = i + 1)
                if (logits[i] > ref_max) begin
                    ref_max = logits[i];
                    ref_idx = i;
                end

            @(negedge clk);
            for (i = 0; i < 47; i = i + 1) begin
                in_wr_en   = 1;
                in_wr_addr = i[5:0];
                in_wr_data = logits[i];
                @(negedge clk);
            end
            in_wr_en = 0;

            wait (ready);
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;

            wait (valid_out);
            #1;
            if (predicted_class !== ref_idx[5:0]) begin
                errors = errors + 1;
                $display("[SAI] test=%0d rtl=%0d ref=%0d", t, predicted_class, ref_idx);
            end
            @(negedge clk);
        end

        if (errors == 0)
            $display("\n[PASS] argmax: tat ca %0d test ngau nhien deu dung.\n", t);
        else
            $display("\n[FAIL] argmax: %0d / %0d test sai.\n", errors, t);

        #20 $finish;
    end

    initial begin
        #200000;
        $display("[WATCHDOG] tb_argmax qua thoi gian cho!");
        $finish;
    end
endmodule