`timescale 1ns / 1ps
// =============================================================
// TB_FC3 : Kiem tra doc lap module fc3 (84 -> 47, khong quant/ReLU)
// =============================================================
module tb_fc3;

    reg clk, rst_n, start;
    wire ready, done;

    reg        in_wr_en;
    reg [6:0]  in_wr_addr;
    reg signed [7:0] in_wr_data;

    reg  [5:0] out_rd_addr;
    wire signed [31:0] out_rd_data;

    fc3 dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .ready(ready), .done(done),
        .in_wr_en(in_wr_en), .in_wr_addr(in_wr_addr), .in_wr_data(in_wr_data),
        .out_rd_addr(out_rd_addr), .out_rd_data(out_rd_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Xuat waveform ra file .vcd de xem bang GTKWave
    initial begin
        $dumpfile("fc3.vcd");
        $dumpvars(0, tb_fc3);
    end

    reg signed [7:0]  ref_kernel [0:3947];
    reg signed [31:0] ref_bias   [0:46];

    reg signed [7:0]  ref_in  [0:83];
    reg signed [31:0] ref_out [0:46];

    integer c, kc, i, errors;
    reg signed [31:0] acc;

    task compute_reference;
        begin
            for (c = 0; c < 47; c = c + 1) begin
                acc = ref_bias[c];
                for (kc = 0; kc < 84; kc = kc + 1)
                    acc = acc + ref_in[kc] * ref_kernel[c*84 + kc];
                ref_out[c] = acc;
            end
        end
    endtask

    initial begin
        $readmemh("weights_hex/fc3_kernel.hex", ref_kernel);
        $readmemh("weights_hex/fc3_bias.hex",   ref_bias);

        rst_n = 0; start = 0; in_wr_en = 0; in_wr_addr = 0; in_wr_data = 0;
        out_rd_addr = 0; errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;

        for (i = 0; i < 84; i = i + 1)
            ref_in[i] = $random;

        compute_reference;

        @(negedge clk);
        for (i = 0; i < 84; i = i + 1) begin
            in_wr_en   = 1;
            in_wr_addr = i[6:0];
            in_wr_data = ref_in[i];
            @(negedge clk);
        end
        in_wr_en = 0;

        wait (ready);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        wait (done);
        @(negedge clk);

        for (i = 0; i < 47; i = i + 1) begin
            out_rd_addr = i[5:0];
            #1;
            if (out_rd_data !== ref_out[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("[SAI] idx=%0d rtl=%0d ref=%0d", i, out_rd_data, ref_out[i]);
            end
        end

        if (errors == 0)
            $display("\n[PASS] fc3: tat ca 47 gia tri logit khop voi mo hinh tham chieu.\n");
        else
            $display("\n[FAIL] fc3: %0d / 47 gia tri sai khac.\n", errors);

        #20 $finish;
    end

    initial begin
        #500000;
        $display("[WATCHDOG] tb_fc3 qua thoi gian cho!");
        $finish;
    end
endmodule