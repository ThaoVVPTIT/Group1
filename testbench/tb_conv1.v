`timescale 1ns / 1ps
// =============================================================
// TB_CONV1 : Kiem tra doc lap module conv1
// Sinh anh 28x28 ngau nhien, chay RTL, doi chieu voi mo hinh
// tham chieu hanh vi (behavioral) tinh cung cong thuc luong tu hoa.
// =============================================================
module tb_conv1;

    reg clk, rst_n;
    reg start;
    wire ready, done;

    reg        in_wr_en;
    reg [9:0]  in_wr_addr;
    reg signed [7:0] in_wr_data;

    reg  [11:0] out_rd_addr;
    wire signed [7:0] out_rd_data;

    conv1 dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .ready(ready), .done(done),
        .in_wr_en(in_wr_en), .in_wr_addr(in_wr_addr), .in_wr_data(in_wr_data),
        .out_rd_addr(out_rd_addr), .out_rd_data(out_rd_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Xuat waveform ra file .vcd de xem bang GTKWave
    initial begin
        $dumpfile("conv1.vcd");
        $dumpvars(0, tb_conv1);
    end

    // Ban sao cua ROM trong so de tinh mo hinh tham chieu trong TB
    reg signed [7:0]  ref_kernel [0:53];
    reg signed [31:0] ref_bias   [0:5];
    reg signed [31:0] ref_mult   [0:5];
    reg [7:0]         ref_shift  [0:5];

    reg signed [7:0] ref_img [0:783];
    reg signed [7:0] ref_out [0:4055];

    integer c, y, x, ky, kx, i, errors;
    reg signed [31:0] acc;
    reg signed [63:0] prod;

    task compute_reference;
        begin
            for (c = 0; c < 6; c = c + 1) begin
                for (y = 0; y < 26; y = y + 1) begin
                    for (x = 0; x < 26; x = x + 1) begin
                        acc = ref_bias[c];
                        for (ky = 0; ky < 3; ky = ky + 1)
                            for (kx = 0; kx < 3; kx = kx + 1)
                                acc = acc + ref_img[(y+ky)*28 + (x+kx)] * ref_kernel[c*9 + ky*3 + kx];

                        prod = acc * ref_mult[c];
                        if (ref_shift[c] > 0) begin
                            prod = prod + (64'sd1 << (ref_shift[c]-1));
                            acc  = prod >>> ref_shift[c];
                        end else acc = prod;

                        if (acc > 127) acc = 127; else if (acc < -128) acc = -128;
                        if (acc < 0) acc = 0;

                        ref_out[c*676 + y*26 + x] = acc[7:0];
                    end
                end
            end
        end
    endtask

    initial begin
        $readmemh("weights_hex/conv1_kernel.hex",     ref_kernel);
        $readmemh("weights_hex/conv1_bias.hex",       ref_bias);
        $readmemh("weights_hex/conv1_multiplier.hex", ref_mult);
        $readmemh("weights_hex/conv1_shift.hex",      ref_shift);

        rst_n = 0; start = 0; in_wr_en = 0; in_wr_addr = 0; in_wr_data = 0;
        out_rd_addr = 0; errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;

        // Sinh anh dau vao ngau nhien
        for (i = 0; i < 784; i = i + 1)
            ref_img[i] = $random;

        compute_reference;

        // Nap anh vao DUT qua cong ghi
        @(negedge clk);
        for (i = 0; i < 784; i = i + 1) begin
            in_wr_en   = 1;
            in_wr_addr = i[9:0];
            in_wr_data = ref_img[i];
            @(negedge clk);
        end
        in_wr_en = 0;

        // Kich chay
        wait (ready);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        wait (done);
        @(negedge clk);

        // So sanh toan bo 4056 gia tri dau ra
        for (i = 0; i < 4056; i = i + 1) begin
            out_rd_addr = i[11:0];
            #1;
            if (out_rd_data !== ref_out[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("[SAI] idx=%0d rtl=%0d ref=%0d", i, out_rd_data, ref_out[i]);
            end
        end

        if (errors == 0)
            $display("\n[PASS] conv1: tat ca 4056 gia tri dau ra khop voi mo hinh tham chieu.\n");
        else
            $display("\n[FAIL] conv1: %0d / 4056 gia tri sai khac.\n", errors);

        #20 $finish;
    end

    initial begin
        #500000;
        $display("[WATCHDOG] tb_conv1 qua thoi gian cho!");
        $finish;
    end
endmodule