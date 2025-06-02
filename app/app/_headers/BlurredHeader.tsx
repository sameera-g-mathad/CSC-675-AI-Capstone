import { StyleSheet, Text, View } from "react-native";
import React from "react";

interface blurredHeaderTitle {
    title: string;
}

export default function BlurredHeader({ title }: blurredHeaderTitle) {
    return (
        <View style={styles.viewStyle}>
            <Text style={styles.textStyle}>{title}</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    viewStyle: {
        height: 50,
        justifyContent: "center",
        alignContent: 'center',
        padding: 10
    },
    textStyle: {
        fontWeight: 'bold',
        fontSize: 20
    }
});
