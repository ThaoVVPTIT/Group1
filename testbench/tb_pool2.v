`timescale 1ns / 1ps
// =============================================================
// TB_POOL2 : Kiem tra doc lap module pool2 (max-pool 2x2)
// =============================================================
module tb_pool2;

    reg clk, rst_n, start;
    wire ready, done;

    reg         in_wr_en;
    reg [10:0]  in_wr_addr;
    reg signed [7:0] in_wr_data;

    reg  [8:0]  out_rd_addr;
    wire signed [7:0] out_rd_data;

    pool2 dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .ready(ready), .done(done),
        .in_wr_en(in_wr_en), .in_wr_addr(in_wr_addr), .in_wr_data(in_wr_data),
        .out_rd_addr(out_rd_addr), .out_rd_data(out_rd_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Xuat waveform ra file .vcd de xem bang GTKWave
    initial begin
        $dumpfile("pool2.vcd");
        $dumpvars(0, tb_pool2);
    end

    reg signed [7:0] ref_in  [0:1935];
    reg signed [7:0] ref_out [0:399];

    integer c, y, x, ky, kx, i, errors;
    reg signed [31:0] max_val;

    task compute_reference;
        begin
            for (c = 0; c < 16; c = c + 1) begin
                for (y = 0; y < 5; y = y + 1) begin
                    for (x = 0; x < 5; x = x + 1) begin
                        max_val = -128;
                        for (ky = 0; ky < 2; ky = ky + 1)
                            for (kx = 0; kx < 2; kx = kx + 1)
                                if (ref_in[c*121 + (y*2+ky)*11 + (x*2+kx)] > max_val)
                                    max_val = ref_in[c*121 + (y*2+ky)*11 + (x*2+kx)];
                        ref_out[c*25 + y*5 + x] = max_val[7:0];
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

        for (i = 0; i < 1936; i = i + 1)
            ref_in[i] = $random;

        compute_reference;

        @(negedge clk);
        for (i = 0; i < 1936; i = i + 1) begin
            in_wr_en   = 1;
            in_wr_addr = i[10:0];
            in_wr_data = ref_in[i];
            @(negedge clk);
        end
        in_wr_en = 0;

        wait (ready);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        wait (done);
        @(negedge clk);

        for (i = 0; i < 400; i = i + 1) begin
            out_rd_addr = i[8:0];
            #1;
            if (out_rd_data !== ref_out[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("[SAI] idx=%0d rtl=%0d ref=%0d", i, out_rd_data, ref_out[i]);
            end
        end

        if (errors == 0)
            $display("\n[PASS] pool2: tat ca 400 gia tri dau ra khop voi mo hinh tham chieu.\n");
        else
            $display("\n[FAIL] pool2: %0d / 400 gia tri sai khac.\n", errors);

        #20 $finish;
    end

    initial begin
        #500000;
        $display("[WATCHDOG] tb_pool2 qua thoi gian cho!");
        $finish;
    end
endmodule