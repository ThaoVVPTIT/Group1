`timescale 1ns / 1ps
// =============================================================
// TB_CONV2 : Kiem tra doc lap module conv2
// =============================================================
module tb_conv2;

    reg clk, rst_n, start;
    wire ready, done;

    reg         in_wr_en;
    reg [9:0]   in_wr_addr;
    reg signed [7:0] in_wr_data;

    reg  [10:0] out_rd_addr;
    wire signed [7:0] out_rd_data;

    conv2 dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .ready(ready), .done(done),
        .in_wr_en(in_wr_en), .in_wr_addr(in_wr_addr), .in_wr_data(in_wr_data),
        .out_rd_addr(out_rd_addr), .out_rd_data(out_rd_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Xuat waveform ra file .vcd de xem bang GTKWave
    initial begin
        $dumpfile("conv2.vcd");
        $dumpvars(0, tb_conv2);
    end

    reg signed [7:0]  ref_kernel [0:863];
    reg signed [31:0] ref_bias   [0:15];
    reg signed [31:0] ref_mult   [0:15];
    reg [7:0]         ref_shift  [0:15];

    reg signed [7:0] ref_in  [0:1013];
    reg signed [7:0] ref_out [0:1935];

    integer c, y, x, ky, kx, kc, i, errors;
    reg signed [31:0] acc;
    reg signed [63:0] prod;

    task compute_reference;
        begin
            for (c = 0; c < 16; c = c + 1) begin
                for (y = 0; y < 11; y = y + 1) begin
                    for (x = 0; x < 11; x = x + 1) begin
                        acc = ref_bias[c];
                        for (kc = 0; kc < 6; kc = kc + 1)
                            for (ky = 0; ky < 3; ky = ky + 1)
                                for (kx = 0; kx < 3; kx = kx + 1)
                                    acc = acc + ref_in[kc*169 + (y+ky)*13 + (x+kx)] * ref_kernel[c*54 + kc*9 + ky*3 + kx];

                        prod = acc * ref_mult[c];
                        if (ref_shift[c] > 0) begin
                            prod = prod + (64'sd1 << (ref_shift[c]-1));
                            acc  = prod >>> ref_shift[c];
                        end else acc = prod;

                        if (acc > 127) acc = 127; else if (acc < -128) acc = -128;
                        if (acc < 0) acc = 0;

                        ref_out[c*121 + y*11 + x] = acc[7:0];
                    end
                end
            end
        end
    endtask

    initial begin
        $readmemh("weights_hex/conv2_kernel.hex",     ref_kernel);
        $readmemh("weights_hex/conv2_bias.hex",       ref_bias);
        $readmemh("weights_hex/conv2_multiplier.hex", ref_mult);
        $readmemh("weights_hex/conv2_shift.hex",      ref_shift);

        rst_n = 0; start = 0; in_wr_en = 0; in_wr_addr = 0; in_wr_data = 0;
        out_rd_addr = 0; errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;

        for (i = 0; i < 1014; i = i + 1)
            ref_in[i] = $random;

        compute_reference;

        @(negedge clk);
        for (i = 0; i < 1014; i = i + 1) begin
            in_wr_en   = 1;
            in_wr_addr = i[9:0];
            in_wr_data = ref_in[i];
            @(negedge clk);
        end
        in_wr_en = 0;

        wait (ready);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        wait (done);
        @(negedge clk);

        for (i = 0; i < 1936; i = i + 1) begin
            out_rd_addr = i[10:0];
            #1;
            if (out_rd_data !== ref_out[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("[SAI] idx=%0d rtl=%0d ref=%0d", i, out_rd_data, ref_out[i]);
            end
        end

        if (errors == 0)
            $display("\n[PASS] conv2: tat ca 1936 gia tri dau ra khop voi mo hinh tham chieu.\n");
        else
            $display("\n[FAIL] conv2: %0d / 1936 gia tri sai khac.\n", errors);

        #20 $finish;
    end

    initial begin
        #2000000;
        $display("[WATCHDOG] tb_conv2 qua thoi gian cho!");
        $finish;
    end
endmodule