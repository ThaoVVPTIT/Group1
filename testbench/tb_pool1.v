`timescale 1ns / 1ps
// =============================================================
// TB_POOL1 : Kiem tra doc lap module pool1 (max-pool 2x2)
// =============================================================
module tb_pool1;

    reg clk, rst_n, start;
    wire ready, done;

    reg         in_wr_en;
    reg [11:0]  in_wr_addr;
    reg signed [7:0] in_wr_data;

    reg  [9:0]  out_rd_addr;
    wire signed [7:0] out_rd_data;

    pool1 dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .ready(ready), .done(done),
        .in_wr_en(in_wr_en), .in_wr_addr(in_wr_addr), .in_wr_data(in_wr_data),
        .out_rd_addr(out_rd_addr), .out_rd_data(out_rd_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Xuat waveform ra file .vcd de xem bang GTKWave
    initial begin
        $dumpfile("pool1.vcd");
        $dumpvars(0, tb_pool1);
    end

    reg signed [7:0] ref_in  [0:4055];
    reg signed [7:0] ref_out [0:1013];

    integer c, y, x, ky, kx, i, errors;
    reg signed [31:0] max_val;

    task compute_reference;
        begin
            for (c = 0; c < 6; c = c + 1) begin
                for (y = 0; y < 13; y = y + 1) begin
                    for (x = 0; x < 13; x = x + 1) begin
                        max_val = -128;
                        for (ky = 0; ky < 2; ky = ky + 1)
                            for (kx = 0; kx < 2; kx = kx + 1)
                                if (ref_in[c*676 + (y*2+ky)*26 + (x*2+kx)] > max_val)
                                    max_val = ref_in[c*676 + (y*2+ky)*26 + (x*2+kx)];
                        ref_out[c*169 + y*13 + x] = max_val[7:0];
                    end
                end
            end
        end
    endtask

    initial begin
        rst_n = 0; start = 0; in_wr_en = 0; in_wr_addr = 0; in_wr_data = 0;
        out_rd_addr = 0; errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;

        for (i = 0; i < 4056; i = i + 1)
            ref_in[i] = $random;

        compute_reference;

        @(negedge clk);
        for (i = 0; i < 4056; i = i + 1) begin
            in_wr_en   = 1;
            in_wr_addr = i[11:0];
            in_wr_data = ref_in[i];
            @(negedge clk);
        end
        in_wr_en = 0;

        wait (ready);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        wait (done);
        @(negedge clk);

        for (i = 0; i < 1014; i = i + 1) begin
            out_rd_addr = i[9:0];
            #1;
            if (out_rd_data !== ref_out[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("[SAI] idx=%0d rtl=%0d ref=%0d", i, out_rd_data, ref_out[i]);
            end
        end

        if (errors == 0)
            $display("\n[PASS] pool1: tat ca 1014 gia tri dau ra khop voi mo hinh tham chieu.\n");
        else
            $display("\n[FAIL] pool1: %0d / 1014 gia tri sai khac.\n", errors);

        #20 $finish;
    end

    initial begin
        #500000;
        $display("[WATCHDOG] tb_pool1 qua thoi gian cho!");
        $finish;
    end
endmodule